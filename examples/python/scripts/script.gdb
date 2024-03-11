set breakpoint pending on

break src/integrator/direct.cpp:65
break src/bsdf/vaesub.cpp:310
break VaeSub::sample
break sample_sub
break src/integrator/integrator.cpp:52 thread 1
break src/scene/scene_optix.cpp:220
break src/scene/scene.cpp:480
break src/bsdf/hetersub.cpp:457
break Scene::configure

run
sbreak __sample_sub<true> thread 1
sbreak __sample_sp<true> thread 1
sbreak __sample_sub<false>  thread 1
sbreak src/scene/scene.cpp:319
sbreak src/bsdf/vaesub.cpp:567
sbreak src/bsdf/vaesub.cpp:380



sbreak src/bsdf/vaesub.cpp:735
sbreak src/bsdf/vaesub.cpp:630
sbreak src/integrator/direct.cpp:71
sbreak __sample_sp<false> thread 1
sbreak __pdf_sub