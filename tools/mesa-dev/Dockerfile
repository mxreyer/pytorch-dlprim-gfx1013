# Throwaway image: Mesa (radeonsi + rusticl only) built from source with the
# ACO experiment knobs from aco-experiments.patch, plus the same torch +
# pytorch_ocl stack build.sh uses, so wino-repro.py can run against a modified
# compiler. Used on 2026-09-10 to rule the compiler OUT as the cause of the
# wrong-gradient bug (OPENCL-PERF.md, Finding 3, "It was the voltage").
# The Mesa build takes ~3 minutes on the BC-250.
#
# Build from the repository root (it copies the pt_ocl.so build.sh left there):
#   docker build -f tools/mesa-dev/Dockerfile -t mesa-dev .
#   docker run --rm --device /dev/dri/renderD128 --group-add "$(getent group render | cut -d: -f3)" \
#     -e MESA_SHADER_CACHE_DISABLE=true -e ACO_FORCE_NOP_MASK=1 \
#     -v $PWD/tools:/t -w /t mesa-dev python wino-repro.py 100
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
FROM registry.fedoraproject.org/fedora:45

# Build deps for Mesa, plus the runtime pieces rusticl needs: libclc (the
# built-in function library, path baked in at build time), ocl-icd (the loader
# pytorch_ocl links against), python3.12 for the venv.
RUN dnf -y install dnf5-plugins && dnf -y builddep mesa \
 && dnf -y install --setopt=install_weak_deps=False \
        libclc ocl-icd clinfo python3.12 xz patch \
 && dnf clean all

# Fedora's libclc-spirv 22.1.8 cannot link sin()/cos()/fma() (an empty
# __clc_flush_denormal_if_not_supported; OPENCL-PERF.md, Finding 8). Mesa's
# fork ships fixed drop-in binaries for the same LLVM.
ARG MESA_LIBCLC_VER=22.1.8.3
ARG MESA_LIBCLC_URL=https://gitlab.freedesktop.org/api/v4/projects/27606/packages/generic/mesa-libclc/${MESA_LIBCLC_VER}
ADD --chmod=0644 --checksum=sha256:728cdfa747a1857829c104291984964971783a301a62e9468a3d5a4f1750c949 \
    ${MESA_LIBCLC_URL}/spirv64-mesa3d-.spv /usr/lib64/clc/spirv64-mesa3d-.spv
ADD --chmod=0644 --checksum=sha256:59552f9e22ec221061e9939fc90a46e4159d1400b2e1f85eff46924489d532aa \
    ${MESA_LIBCLC_URL}/spirv-mesa3d-.spv   /usr/lib64/clc/spirv-mesa3d-.spv

# Mesa from source, radeonsi + rusticl only, into /opt/mesa.
WORKDIR /src
ADD https://archive.mesa3d.org/mesa-26.2.2.tar.xz /src/
RUN tar xJf mesa-26.2.2.tar.xz && rm mesa-26.2.2.tar.xz
COPY tools/mesa-dev/aco-experiments.patch /src/
RUN cd mesa-26.2.2 && patch -p1 < ../aco-experiments.patch
RUN cd mesa-26.2.2 && meson setup build --prefix=/opt/mesa --libdir=lib64 --buildtype=release \
      -Dgallium-drivers=radeonsi -Dvulkan-drivers= -Dplatforms= -Dglx=disabled -Degl=disabled \
      -Dgles1=disabled -Dgles2=disabled -Dopengl=false -Dgbm=disabled -Dllvm=enabled -Dshared-llvm=enabled \
      -Dgallium-rusticl=true -Dgallium-va=disabled -Dvideo-codecs= \
      -Dtools= -Dbuild-tests=false -Dvalgrind=disabled -Dlibunwind=disabled -Dzstd=enabled \
    && ninja -C build -j6 && ninja -C build install
# ICD pointing at the built rusticl (gallium is linked in statically with
# -Dopengl=false, so nothing else is needed).
RUN mkdir -p /opt/mesa/icd && echo /opt/mesa/lib64/libRusticlOpenCL.so.1 > /opt/mesa/icd/rusticl.icd

# Same torch + pytorch_ocl as build.sh, with the patched pt_ocl.so swapped in.
ENV VENV=/opt/venv PATH=/opt/venv/bin:/usr/bin:/bin \
    OCL_ICD_VENDORS=/opt/mesa/icd RUSTICL_ENABLE=radeonsi
RUN python3.12 -m venv $VENV \
 && pip install --no-cache-dir -U pip \
 && pip install --no-cache-dir torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir \
        "https://github.com/artyom-beilis/pytorch_dlprim/releases/download/0.2.0/pytorch_ocl-0.2.0+torch2.4-cp312-none-linux_x86_64.whl"
COPY --chmod=0644 pt_ocl.so $VENV/lib/python3.12/site-packages/pytorch_ocl/pt_ocl.so

# Unprivileged user; /dev/dri/renderD128 access comes from --group-add at run time.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user
