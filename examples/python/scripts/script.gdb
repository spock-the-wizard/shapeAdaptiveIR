set breakpoint pending on

break src/integrator/direct.cpp:65

run
sbreak __sample_sub<true> thread 1
sbreak __sample_sp<true> thread 1
sbreak __sample_sub<false>  thread 1
sbreak src/scene/scene.cpp:319
sbreak src/bsdf/vaesub.cpp:567
sbreak src/bsdf/vaesub.cpp:564
sbreak src/bsdf/vaesub.cpp:380
sbreak src/bsdf/vaesub.cpp:735
sbreak src/bsdf/vaesub.cpp:630
sbreak src/integrator/direct.cpp:71
sbreak __sample_sp<false> thread 1
sbreak __pdf_sub