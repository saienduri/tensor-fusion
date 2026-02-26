package utils_test

import (
	"context"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/utils"
)

var _ = Describe("Compose Utils", func() {
	Describe("AddTFHypervisorConfAfterTemplate", func() {
		DescribeTable("configures hypervisor correctly",
			func(enableVector bool, hypervisorImage string, expectInitCount, expectVolumeCount int) {
				ctx := context.Background()
				spec := &corev1.PodSpec{}
				pool := &tfv1.GPUPool{
					Spec: tfv1.GPUPoolSpec{
						ComponentConfig: &tfv1.ComponentConfig{
							Hypervisor: &tfv1.HypervisorConfig{
								Image:        hypervisorImage,
								EnableVector: enableVector,
							},
						},
					},
				}

				utils.AddTFHypervisorConfAfterTemplate(ctx, spec, pool, "NVIDIA", false)

				Expect(spec.InitContainers).To(HaveLen(expectInitCount), "unexpected number of init containers")
				Expect(spec.Volumes).To(HaveLen(expectVolumeCount), "unexpected number of volumes")
				Expect(spec.HostPID).To(BeTrue())
				Expect(spec.TerminationGracePeriodSeconds).NotTo(BeNil())
			},
			Entry("without vector", false, "test-image:latest", 2, 7),
			Entry("with vector", true, "test-image:latest", 2, 7),
		)
	})

	Describe("SetWorkerContainerSpec", func() {
		DescribeTable("configures worker container correctly",
			func(gpuVendor, workerImage, disabledFeatures string, sharedMemMode bool, expectCommand []string) {
				container := &corev1.Container{}
				workloadProfile := &tfv1.WorkloadProfileSpec{GPUVendor: gpuVendor}
				workerConfig := &tfv1.WorkerConfig{
					Image: workerImage,
				}
				hypervisorConfig := &tfv1.HypervisorConfig{}

				utils.SetWorkerContainerSpec(container, workloadProfile, workerConfig, hypervisorConfig, disabledFeatures, sharedMemMode)

				Expect(container.Name).NotTo(BeEmpty())
				if workerImage != "" {
					Expect(container.Image).To(Equal(workerImage))
				}

				// Verify command is set correctly
				if expectCommand != nil {
					Expect(container.Command).To(Equal(expectCommand), "container command should match expected value")
				} else {
					Expect(container.Command).To(BeNil())
				}

				// Verify shared memory mode specific setup
				if sharedMemMode && disabledFeatures == "" {
					Expect(container.Command).To(HaveLen(3), "shared memory mode should use bash -c with script")
					Expect(container.Command[0]).To(Equal("/bin/bash"), "should use bash")
					Expect(container.Command[1]).To(Equal("-c"), "should use -c flag")
					Expect(container.Command[2]).To(ContainSubstring("touch /dev/shm/tf_shm"), "should create shared memory file")
					Expect(container.Command[2]).To(ContainSubstring("chmod 666 /dev/shm/tf_shm"), "should set file permissions")
					Expect(container.Command[2]).To(ContainSubstring("exec ./tensor-fusion-worker"), "should exec worker")
					Expect(container.Command[2]).To(ContainSubstring("-n shmem"), "should use shmem mode")
					Expect(container.Command[2]).To(ContainSubstring("-m tf_shm"), "should specify shared memory name")
					Expect(container.Command[2]).To(ContainSubstring("-M 256"), "should specify shared memory size")
				}
			},
			Entry("basic worker config", "", "worker:latest", "", false, []string{
				"./tensor-fusion-worker",
				"-p",
				"8000",
			}),
			Entry("NVIDIA worker config", "NVIDIA", "worker:latest", "", false, []string{
				"./tensor-fusion-worker",
				"-p",
				"8000",
			}),
			Entry("AMD worker uses entrypoint", "AMD", "worker:latest", "", false, nil),
			Entry("worker with shared memory mode", "", "worker:latest", "", true, []string{
				"/bin/bash",
				"-c",
				"touch /dev/shm/tf_shm && chmod 666 /dev/shm/tf_shm && exec ./tensor-fusion-worker -n shmem -m tf_shm -M 256",
			}),
			Entry("worker with disabled start-worker feature", "", "worker:latest", "start-worker", false, []string{
				"sleep",
				"infinity",
			}),
		)
	})

	Describe("SetWorkerContainerSpec LD_PRELOAD", func() {
		DescribeTable("sets correct limiter path per vendor",
			func(gpuVendor, disabledFeatures, expectLdPreload string, expectNoLdPreload bool) {
				container := &corev1.Container{}
				workloadProfile := &tfv1.WorkloadProfileSpec{GPUVendor: gpuVendor}
				workerConfig := &tfv1.WorkerConfig{Image: "test:latest"}
				hypervisorConfig := &tfv1.HypervisorConfig{}

				utils.SetWorkerContainerSpec(container, workloadProfile, workerConfig, hypervisorConfig, disabledFeatures, false)

				ldPreload := findEnv(container.Env, "LD_PRELOAD")
				if expectNoLdPreload {
					Expect(ldPreload).To(BeNil())
				} else {
					Expect(ldPreload).NotTo(BeNil())
					Expect(ldPreload.Value).To(Equal(expectLdPreload))
				}
			},
			Entry("NVIDIA gets cuda limiter", "NVIDIA", "", "/home/app/libcuda_limiter.so", false),
			Entry("AMD gets hip limiter", "AMD", "", "/usr/lib/tensor-fusion/libhip_limiter.so", false),
			Entry("default gets cuda limiter", "", "", "/home/app/libcuda_limiter.so", false),
			Entry("NVIDIA with gpu-limiter disabled", "NVIDIA", "gpu-limiter", "", true),
			Entry("AMD with gpu-limiter disabled", "AMD", "gpu-limiter", "", true),
		)
	})
})

func findEnv(envs []corev1.EnvVar, name string) *corev1.EnvVar {
	for i := range envs {
		if envs[i].Name == name {
			return &envs[i]
		}
	}
	return nil
}
