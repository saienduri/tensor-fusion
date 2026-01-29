/*
Copyright 2024.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package v1

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type ResourceName string

const (
	ResourceTflops ResourceName = "tflops"
	ResourceVram   ResourceName = "vram"
)

type Resource struct {
	Tflops resource.Quantity `json:"tflops"`

	// +optional
	// 0-100 percentage, mutually exclusive with TFLOPs
	ComputePercent resource.Quantity `json:"compute,omitempty"`

	Vram resource.Quantity `json:"vram"`
}

type Resources struct {
	Requests Resource `json:"requests"`
	Limits   Resource `json:"limits"`
}

func (r Resources) Equal(target *Resources) bool {
	if target == nil {
		return false
	}
	return r.Requests.Tflops.Equal(target.Requests.Tflops) &&
		r.Requests.Vram.Equal(target.Requests.Vram) &&
		r.Limits.Tflops.Equal(target.Limits.Tflops) &&
		r.Limits.Vram.Equal(target.Limits.Vram)
}

func (r Resources) IsZero() bool {
	return r.Requests.Tflops.IsZero() &&
		r.Requests.Vram.IsZero() &&
		r.Limits.Tflops.IsZero() &&
		r.Limits.Vram.IsZero()
}

// ExternalClientSpec defines configuration for external (non-Kubernetes) clients
type ExternalClientSpec struct {
	// Unique identifier for the external client
	// +optional
	ClientID string `json:"clientId,omitempty"`

	// TTL in seconds - connection will be garbage collected after this duration
	// +kubebuilder:validation:Minimum=60
	// +kubebuilder:validation:Maximum=86400
	TTLSeconds int64 `json:"ttlSeconds"`
}

// TensorFusionConnectionSpec defines the desired state of TensorFusionConnection.
type TensorFusionConnectionSpec struct {
	WorkloadName string `json:"workloadName"`

	// ClientPod is the name of the client pod (for in-cluster clients)
	// +optional
	ClientPod string `json:"clientPod,omitempty"`

	// ExternalClient configuration for clients outside the Kubernetes cluster
	// Mutually exclusive with ClientPod
	// +optional
	ExternalClient *ExternalClientSpec `json:"externalClient,omitempty"`
}

// IsExternal returns true if this connection is for an external client
func (s *TensorFusionConnectionSpec) IsExternal() bool {
	return s.ExternalClient != nil
}

// TensorFusionConnectionStatus defines the observed state of TensorFusionConnection.
type TensorFusionConnectionStatus struct {
	Phase         WorkerPhase `json:"phase"`
	ConnectionURL string      `json:"connectionURL"`
	WorkerName    string      `json:"workerName"`

	// ExpiresAt is when this connection expires (for external clients with TTL)
	// +optional
	ExpiresAt *metav1.Time `json:"expiresAt,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Connection URL",type="string",JSONPath=".status.connectionURL"
// +kubebuilder:printcolumn:name="Worker Name",type="string",JSONPath=".status.workerName"
// +kubebuilder:printcolumn:name="Workload Name",type="string",JSONPath=".spec.workloadName"
// +kubebuilder:printcolumn:name="Client Pod",type="string",JSONPath=".spec.clientPod",priority=1
// +kubebuilder:printcolumn:name="External",type="boolean",JSONPath=".spec.externalClient",priority=1
// +kubebuilder:printcolumn:name="Expires At",type="date",JSONPath=".status.expiresAt",priority=1

// TensorFusionConnection is the Schema for the tensorfusionconnections API.
type TensorFusionConnection struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   TensorFusionConnectionSpec   `json:"spec,omitempty"`
	Status TensorFusionConnectionStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// TensorFusionConnectionList contains a list of TensorFusionConnection.
type TensorFusionConnectionList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []TensorFusionConnection `json:"items"`
}

func init() {
	SchemeBuilder.Register(&TensorFusionConnection{}, &TensorFusionConnectionList{})
}
