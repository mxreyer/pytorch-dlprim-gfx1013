# Throwaway image: Mesa (radeonsi + rusticl only) built from source with the ACO
# experiment knobs in aco-experiments.patch, on top of the notebook image so the
# same torch/pytorch_ocl stack runs against it. Used on 2026-09-10 to rule the
# compiler OUT as the cause of the wrong-gradient bug (OPENCL-PERF.md, Finding 3,
# "It was the voltage"). Build takes ~3 minutes on the BC-250.
#
#   docker build -t mesa-dev tools/mesa-dev
#   docker run --rm --device /dev/dri/renderD128 --group-add 105 \
#     -e OCL_ICD_VENDORS=/opt/mesa/icd -e MESA_SHADER_CACHE_DISABLE=true \
#     -e ACO_FORCE_NOP_MASK=1 -v $PWD/tools:/t -w /t mesa-dev python wino-repro.py 100
#
# Knobs the patch adds (all read at shader compile time; keep the cache off):
#   ACO_LDS_BARRIER_WAIT=1        s_waitcnt lgkmcnt(0) instead of vm_vsrc(0) before
#                                 control barriers on GFX10+ CU mode
#   ACO_DEBUG=force-waitdeps + ACO_FORCE_DEPCTR_IMM=0x....
#                                 choose the s_waitcnt_depctr immediate the forced
#                                 wait uses (0xffff = a wait for nothing)
#   ACO_FORCE_NOP_MASK=<bits> [ACO_FORCE_NOP_N=<n>]
#                                 s_nop n before every instruction of the classes
#                                 1 VALU 2 SALU 4 VMEM 8 DS 16 SMEM 32 SOPP 64 other
#   ACO_FORCE_NOP_AFTER_WAITCNT=1 s_nop 0 after every s_waitcnt
# None of these is a fix for anything; they exist to show that ANY extra issue
# slot made the failure vanish, which is what pointed at the voltage.
FROM bc250-notebook:f45
USER root
RUN dnf -y install dnf5-plugins && dnf -y builddep mesa && dnf -y install xz patch && dnf clean all
WORKDIR /src
ADD https://archive.mesa3d.org/mesa-26.2.2.tar.xz /src/
RUN tar xJf mesa-26.2.2.tar.xz && rm mesa-26.2.2.tar.xz
COPY aco-experiments.patch /src/
RUN cd mesa-26.2.2 && patch -p1 < ../aco-experiments.patch
RUN cd mesa-26.2.2 && meson setup build --prefix=/opt/mesa --libdir=lib64 --buildtype=release \
      -Dgallium-drivers=radeonsi -Dvulkan-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled \
      -Dgles1=disabled -Dgles2=disabled -Dopengl=false -Dgbm=disabled -Dllvm=enabled -Dshared-llvm=enabled \
      -Dgallium-rusticl=true -Dgallium-va=disabled -Dvideo-codecs= \
      -Dtools= -Dbuild-tests=false -Dvalgrind=disabled -Dlibunwind=disabled -Dzstd=enabled \
    && ninja -C build -j6 && ninja -C build install
# ICD pointing at the built rusticl (gallium is linked in statically with
# -Dopengl=false, so nothing else is needed); select with OCL_ICD_VENDORS=/opt/mesa/icd
RUN mkdir -p /opt/mesa/icd && echo /opt/mesa/lib64/libRusticlOpenCL.so.1 > /opt/mesa/icd/rusticl.icd
USER jovyan
WORKDIR /home/jovyan
