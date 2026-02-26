#!/bin/bash
# Copyright 2024.
# TensorFusion Init Container Entrypoint
# Sets up GPU interception for different isolation modes:
# - shared: No interception, direct GPU access
# - remote: HIP/CUDA API forwarding over network (GPU-over-IP)
# - soft/hard: Local interception with resource limiting (AMD: hip-limiter)

set -e

echo "TensorFusion: Starting init for ${HARDWARE_VENDOR:-UNKNOWN}"

# Directories
LIBS_DIR="${TF_LIBS_DIR:-/tensor-fusion}"
CONF_DIR="${TF_CONF_DIR:-/tensor-fusion-conf}"

# Create target directories
mkdir -p "${LIBS_DIR}"
mkdir -p "${CONF_DIR}"

# Get isolation mode from environment (set by webhook)
# remote mode is triggered when IS_LOCAL_GPU=false
IS_LOCAL_GPU="${IS_LOCAL_GPU:-true}"
ISOLATION_MODE="${TF_ISOLATION_MODE:-shared}"

echo "Hardware vendor: ${HARDWARE_VENDOR}"
echo "Local GPU: ${IS_LOCAL_GPU}"
echo "Isolation mode: ${ISOLATION_MODE}"

# Check for remote GPU mode (non-local GPU)
if [ "${IS_LOCAL_GPU}" = "false" ]; then
    echo "Remote GPU mode detected - setting up API forwarding"
    
    if [ "${HARDWARE_VENDOR}" = "AMD" ]; then
        # AMD remote mode: Use HIP client stub for API forwarding
        HIP_CLIENT_STUB="/build/lib/libhip_client_stub.so"
        
        if [ -f "${HIP_CLIENT_STUB}" ]; then
            echo "Copying HIP client stub for remote GPU access"
            cp "${HIP_CLIENT_STUB}" "${LIBS_DIR}/"
            
            # Configure LD_PRELOAD to intercept HIP calls
            echo "${LIBS_DIR}/libhip_client_stub.so" > "${CONF_DIR}/ld.so.preload"
            
            # Create library path config
            echo "${LIBS_DIR}" > "${CONF_DIR}/zz_tensor-fusion.conf"
            
            echo "HIP client stub installed at ${LIBS_DIR}/libhip_client_stub.so"
            echo "LD_PRELOAD configured to intercept HIP API calls"
        else
            echo "WARNING: HIP client stub not found at ${HIP_CLIENT_STUB}"
            echo "Remote GPU mode will not work properly"
            # Create empty files to prevent mount failures
            touch "${CONF_DIR}/ld.so.preload"
            touch "${CONF_DIR}/zz_tensor-fusion.conf"
        fi
    else
        echo "WARNING: Unknown vendor ${HARDWARE_VENDOR} for remote mode"
        touch "${CONF_DIR}/ld.so.preload"
        touch "${CONF_DIR}/zz_tensor-fusion.conf"
    fi
    
    echo "TensorFusion: Init complete (remote GPU mode)"
    
elif [ "${ISOLATION_MODE}" = "shared" ]; then
    # Shared mode: No interception needed, application uses GPU directly
    echo "Shared isolation mode - no GPU call interception needed"
    echo "Application will use native ROCm/CUDA libraries directly"
    
    # Create empty config files so mounts don't fail
    touch "${CONF_DIR}/ld.so.preload"
    touch "${CONF_DIR}/zz_tensor-fusion.conf"
    
    echo "TensorFusion: Init complete (pass-through mode)"
    
else
    # Soft/Hard isolation modes: Intercept GPU calls with vendor-specific limiter
    echo "Setting up ${ISOLATION_MODE} isolation for ${HARDWARE_VENDOR}"

    if [ "${HARDWARE_VENDOR}" = "AMD" ]; then
        HIP_LIMITER="/usr/lib/tensor-fusion/libhip_limiter.so"

        if [ -f "${HIP_LIMITER}" ]; then
            echo "Copying HIP limiter for memory enforcement"
            cp "${HIP_LIMITER}" "${LIBS_DIR}/"

            # Configure ld.so.preload so glibc intercepts HIP calls at process startup
            echo "${LIBS_DIR}/libhip_limiter.so" > "${CONF_DIR}/ld.so.preload"

            # Add library directory to linker search path
            echo "${LIBS_DIR}" > "${CONF_DIR}/zz_tensor-fusion.conf"

            echo "HIP limiter staged for ${ISOLATION_MODE} isolation"
        else
            echo "WARNING: HIP limiter not found at ${HIP_LIMITER}"
            touch "${CONF_DIR}/ld.so.preload"
            touch "${CONF_DIR}/zz_tensor-fusion.conf"
        fi
    else
        # NVIDIA path handled by bootstrap binary, not this script
        touch "${CONF_DIR}/ld.so.preload"
        touch "${CONF_DIR}/zz_tensor-fusion.conf"
    fi

    echo "TensorFusion: Init complete (${ISOLATION_MODE} isolation)"
fi

