GPU_ARCH ?= gfx1201
HIPCC ?= hipcc
HIPFLAGS = --offload-arch=$(GPU_ARCH) -O3

all: coalesce_bench

coalesce_bench: coalesce_bench.hip
	$(HIPCC) $(HIPFLAGS) -o $@ $<

run: coalesce_bench
	./coalesce_bench

clean:
	rm -f coalesce_bench

.PHONY: all run clean
