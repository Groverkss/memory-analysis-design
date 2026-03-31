#!/usr/bin/env python3
"""Convert dispatch-interior MLIR (with hal bindings) to module-level MLIR.

Reads a dispatch-interior .mlir file that uses hal.interface.binding.subspan
and iree_tensor_ext.dispatch.tensor.load/store, and produces a module-level
.mlir file with func.func arguments and return values.

Skips kernels with dynamic shapes (contains '?').
"""

import re
import sys
import json
from pathlib import Path

# Match: hal.interface.binding.subspan ... binding(N) ... <readonly:tensor<SHAPE>> or <writeonly:tensor<SHAPE>>
BINDING_RE = re.compile(
    r'%(\w+)\s*=\s*hal\.interface\.binding\.subspan\s+layout\([^)]+\)\s+'
    r'binding\((\d+)\).*?<(readonly|writeonly):tensor<([^>]+)>>'
)

# Match: dispatch.tensor.load %binding, ... -> tensor<SHAPE>
LOAD_RE = re.compile(
    r'%(\w+)\s*=\s*iree_tensor_ext\.dispatch\.tensor\.load\s+%(\w+).*?->\s*tensor<([^>]+)>'
)

# Match: dispatch.tensor.store %val, %binding, ...
# val can be %name or %name#N for multi-result ops.
STORE_RE = re.compile(
    r'iree_tensor_ext\.dispatch\.tensor\.store\s+%(\w+(?:#\d+)?),\s*%(\w+)'
)

# Match: func.func @name(...)
FUNC_RE = re.compile(r'func\.func\s+@(\w+)\s*\(')

# Size of element types in bytes.
ELEM_BYTES = {
    'f16': 2, 'bf16': 2, 'f32': 4, 'f64': 8,
    'f8E4M3FNUZ': 1, 'f8E5M2FNUZ': 1, 'f8E4M3FN': 1, 'f8E5M2': 1,
    'i8': 1, 'i16': 2, 'i32': 4, 'i64': 8, 'i1': 1,
}


def parse_shape(shape_str):
    """Parse 'NxMxf16' into (dims_list, element_type)."""
    parts = shape_str.split('x')
    # Last part is the element type.
    elem_type = parts[-1]
    dims = parts[:-1]
    return dims, elem_type


def compute_bytes(shape_str):
    """Compute total bytes for a tensor shape like '2048x4096xf16'."""
    dims, elem_type = parse_shape(shape_str)
    if any(d == '?' for d in dims):
        return None
    total_elems = 1
    for d in dims:
        total_elems *= int(d)
    return total_elems * ELEM_BYTES.get(elem_type, 4)


def convert(input_path):
    """Convert dispatch-interior MLIR to module-level MLIR.

    Returns (module_mlir, metadata) or (None, None) if conversion fails.
    metadata = {name, inputs: [{shape, type}], outputs: [{shape, type}], bytes_moved}
    """
    text = Path(input_path).read_text()

    # Skip dynamic kernels.
    if '?x' in text or '?>' in text:
        return None, None

    # Extract function name.
    func_match = FUNC_RE.search(text)
    if not func_match:
        return None, None
    func_name = func_match.group(1)

    # Extract bindings: {binding_idx: (var_name, mode, shape_str)}
    bindings = {}
    for m in BINDING_RE.finditer(text):
        var_name, binding_idx, mode, shape_str = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        bindings[binding_idx] = (var_name, mode, shape_str)

    if not bindings:
        return None, None

    # Separate inputs and outputs.
    inputs = sorted([(idx, b) for idx, b in bindings.items() if b[1] == 'readonly'])
    outputs = sorted([(idx, b) for idx, b in bindings.items() if b[1] == 'writeonly'])

    if not outputs:
        return None, None

    # Extract loads: {loaded_var: (binding_var, shape)}
    loads = {}
    for m in LOAD_RE.finditer(text):
        loaded_var, binding_var, shape = m.group(1), m.group(2), m.group(3)
        loads[loaded_var] = (binding_var, shape)

    # Extract stores: {binding_var: stored_val}
    stores = {}
    for m in STORE_RE.finditer(text):
        stored_val, binding_var = m.group(1), m.group(2)
        stores[binding_var] = stored_val

    # Build function signature.
    arg_types = []
    arg_names = []
    for idx, (var_name, mode, shape_str) in inputs:
        arg_types.append(f'tensor<{shape_str}>')
        arg_names.append(var_name)

    ret_types = []
    for idx, (var_name, mode, shape_str) in outputs:
        ret_types.append(f'tensor<{shape_str}>')

    # Build the function body by transforming the original.
    lines = text.split('\n')
    body_lines = []
    in_func = False
    brace_depth = 0
    skip_next_return = False

    # Map: binding_var -> arg index (for input bindings)
    binding_to_arg = {}
    for i, (idx, (var_name, mode, shape_str)) in enumerate(inputs):
        binding_to_arg[var_name] = i

    # Map: loaded_var -> arg reference
    load_to_arg = {}
    for loaded_var, (binding_var, shape) in loads.items():
        if binding_var in binding_to_arg:
            load_to_arg[loaded_var] = f'%arg{binding_to_arg[binding_var]}'

    # Map: binding_var (output) -> stored value
    output_binding_to_val = {}
    for idx, (var_name, mode, shape_str) in outputs:
        if var_name in stores:
            output_binding_to_val[var_name] = stores[var_name]

    # Now rebuild the IR.
    output_lines = []
    # Module header.
    output_lines.append('module {')

    # Function signature.
    args_str = ', '.join(f'%arg{i}: {t}' for i, t in enumerate(arg_types))
    rets_str = ', '.join(ret_types)
    if rets_str:
        output_lines.append(f'  func.func @{func_name}({args_str}) -> ({rets_str}) {{')
    else:
        output_lines.append(f'  func.func @{func_name}({args_str}) {{')

    # Process body lines: skip pipeline_layout, binding, load, store lines.
    # Replace references to loaded vars with arg references.
    in_func_body = False
    func_brace_depth = 0

    for line in lines:
        stripped = line.strip()

        # Skip pipeline_layout definition.
        if '#pipeline_layout' in stripped:
            continue

        # Skip blank lines before func.
        if not in_func_body and not stripped:
            continue

        # Detect function start.
        if FUNC_RE.search(stripped):
            in_func_body = True
            func_brace_depth = 1
            continue

        if not in_func_body:
            continue

        # Track brace depth.
        func_brace_depth += stripped.count('{') - stripped.count('}')

        # End of function.
        if func_brace_depth <= 0:
            break

        # Skip binding subspan lines.
        if 'hal.interface.binding.subspan' in stripped:
            continue

        # Skip dispatch.tensor.load lines.
        if 'iree_tensor_ext.dispatch.tensor.load' in stripped:
            continue

        # Skip dispatch.tensor.store lines.
        if 'iree_tensor_ext.dispatch.tensor.store' in stripped:
            continue

        # Skip original return.
        if stripped == 'return':
            continue

        # Replace references to loaded variables with arg references.
        modified = line
        for loaded_var, arg_ref in load_to_arg.items():
            # Replace %loaded_var with the arg reference, being careful about
            # word boundaries.
            modified = re.sub(r'%' + re.escape(loaded_var) + r'\b', arg_ref, modified)

        output_lines.append('  ' + modified.rstrip())

    # Add return statement with output values.
    ret_vals = []
    for idx, (var_name, mode, shape_str) in outputs:
        if var_name in stores:
            val = '%' + stores[var_name]
            # Apply load-to-arg substitutions to the return value too.
            for loaded_var, arg_ref in load_to_arg.items():
                val = re.sub(r'%' + re.escape(loaded_var) + r'\b', arg_ref, val)
            ret_vals.append(val)

    if ret_vals:
        ret_vals_str = ', '.join(ret_vals)
        ret_types_str = ', '.join(ret_types)
        output_lines.append(f'    return {ret_vals_str} : {ret_types_str}')

    output_lines.append('  }')
    output_lines.append('}')

    module_mlir = '\n'.join(output_lines) + '\n'

    # Compute metadata.
    bytes_moved = 0
    input_meta = []
    for idx, (var_name, mode, shape_str) in inputs:
        b = compute_bytes(shape_str)
        if b:
            bytes_moved += b
        input_meta.append({'shape': shape_str, 'bytes': b})

    output_meta = []
    for idx, (var_name, mode, shape_str) in outputs:
        b = compute_bytes(shape_str)
        if b:
            bytes_moved += b
        output_meta.append({'shape': shape_str, 'bytes': b})

    metadata = {
        'name': func_name,
        'inputs': input_meta,
        'outputs': output_meta,
        'bytes_moved': bytes_moved,
    }

    return module_mlir, metadata


def main():
    if len(sys.argv) < 3:
        print(f'Usage: {sys.argv[0]} <input.mlir> <output.mlir> [metadata.json]')
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    meta_path = sys.argv[3] if len(sys.argv) > 3 else None

    mlir, meta = convert(input_path)
    if mlir is None:
        print(f'SKIP: {input_path} (dynamic shapes or parse failure)')
        sys.exit(2)

    Path(output_path).write_text(mlir)
    if meta_path:
        Path(meta_path).write_text(json.dumps(meta, indent=2) + '\n')

    print(f'OK: {meta["name"]} ({len(meta["inputs"])} inputs, {len(meta["outputs"])} outputs, {meta["bytes_moved"]} bytes)')


if __name__ == '__main__':
    main()
