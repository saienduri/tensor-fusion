package router

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"time"

	tfv1 "github.com/NexusGPU/tensor-fusion/api/v1"
	"github.com/NexusGPU/tensor-fusion/internal/constants"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"golang.org/x/crypto/bcrypt"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apimachineryruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/gin-gonic/gin"
)

func mustBcryptHash(raw string) string {
	h, err := bcrypt.GenerateFromPassword([]byte(raw), 10)
	Expect(err).ToNot(HaveOccurred())
	return string(h)
}

func newTestScheme() *apimachineryruntime.Scheme {
	s := apimachineryruntime.NewScheme()
	Expect(corev1.AddToScheme(s)).To(Succeed())
	Expect(tfv1.AddToScheme(s)).To(Succeed())
	return s
}

func newTestSecret(apiKeyPlain string, namespaces []string) *corev1.Secret {
	keys := map[string]*APIKeyInfo{
		"test-key": {
			Hash:       mustBcryptHash(apiKeyPlain),
			Owner:      "unit-test",
			Namespaces:  namespaces,
			CreatedAt:   time.Now().Format(time.RFC3339),
			MaxConnections: 0,
		},
	}
	raw, err := json.Marshal(keys)
	Expect(err).ToNot(HaveOccurred())

	return &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      constants.ExternalConnectionSecretName,
			Namespace: constants.ExternalConnectionSecretNamespace,
		},
		Data: map[string][]byte{
			constants.ExternalConnectionSecretDataKey: raw,
		},
	}
}

func newCreateCtx(body any, apiKey string) (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	b, err := json.Marshal(body)
	Expect(err).ToNot(HaveOccurred())
	req := httptest.NewRequest(http.MethodPost, "/api/v1/external-connections", bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set(constants.ExternalConnectionAPIKeyHeader, apiKey)
	}
	ctx.Request = req
	return ctx, w
}

func newParamCtx(method, path, apiKey string, params gin.Params) (*gin.Context, *httptest.ResponseRecorder) {
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	ctx, _ := gin.CreateTestContext(w)
	req := httptest.NewRequest(method, path, nil)
	if apiKey != "" {
		req.Header.Set(constants.ExternalConnectionAPIKeyHeader, apiKey)
	}
	ctx.Request = req
	ctx.Params = params
	return ctx, w
}

var _ = Describe("ExternalConnectionRouter", func() {
	var (
		scheme *apimachineryruntime.Scheme
		k8s    client.Client
		r      *ExternalConnectionRouter
	)

	BeforeEach(func() {
		scheme = newTestScheme()
	})

	Context("API key auth", func() {
		It("rejects missing API key", func() {
			k8s = fake.NewClientBuilder().
				WithScheme(scheme).
				Build()
			r = NewExternalConnectionRouter(k8s)

			ctx, w := newCreateCtx(map[string]any{
				"poolName":    "pool",
				"namespace":   "default",
				"ttlSeconds":  3600,
				"resources":   map[string]any{"vramRequest": "4Gi", "vramLimit": "4Gi"},
				"clientId":    "c1",
			}, "")

			r.Create(ctx)
			Expect(w.Code).To(Equal(http.StatusUnauthorized))
		})
	})

	Context("Create external connection", func() {
		It("creates a connection and an owned workload when workloadName is omitted", func() {
			apiKey := "rre_unit_test_key"

			pool := &tfv1.GPUPool{
				ObjectMeta: metav1.ObjectMeta{Name: "amd-remote-cluster-amd-remote-pool"},
				Spec: tfv1.GPUPoolSpec{
					NodeManagerConfig: &tfv1.NodeManagerConfig{DefaultVendor: "AMD"},
					CapacityConfig:    &tfv1.CapacityConfig{},
					ComponentConfig:   &tfv1.ComponentConfig{},
					QosConfig:         &tfv1.QosConfig{},
				},
			}

			secret := newTestSecret(apiKey, []string{"default"})

			k8s = fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(pool, secret).
				WithStatusSubresource(&tfv1.TensorFusionConnection{}).
				Build()
			r = NewExternalConnectionRouter(k8s)

			ctx, w := newCreateCtx(map[string]any{
				"poolName":   pool.Name,
				"namespace":  "default",
				"clientId":   "my-laptop",
				"ttlSeconds": 3600,
				"resources": map[string]any{
					"vramRequest": "4Gi",
					"vramLimit":   "4Gi",
				},
			}, apiKey)

			r.Create(ctx)
			Expect(w.Code).To(Equal(http.StatusCreated))

			var resp ExternalConnectionResponse
			Expect(json.Unmarshal(w.Body.Bytes(), &resp)).To(Succeed())
			Expect(resp.Name).To(HavePrefix("ext-"))
			Expect(resp.Namespace).To(Equal("default"))
			Expect(resp.Status).ToNot(BeEmpty())

			createdConn := &tfv1.TensorFusionConnection{}
			Expect(k8s.Get(context.Background(), types.NamespacedName{Name: resp.Name, Namespace: "default"}, createdConn)).To(Succeed())
			Expect(createdConn.Spec.ExternalClient).ToNot(BeNil())
			Expect(createdConn.Spec.WorkloadName).ToNot(BeEmpty())

			createdWorkload := &tfv1.TensorFusionWorkload{}
			Expect(k8s.Get(context.Background(), types.NamespacedName{Name: createdConn.Spec.WorkloadName, Namespace: "default"}, createdWorkload)).To(Succeed())
			Expect(createdWorkload.OwnerReferences).ToNot(BeEmpty())
			Expect(createdWorkload.OwnerReferences[0].Kind).To(Equal("TensorFusionConnection"))
			Expect(createdWorkload.OwnerReferences[0].Name).To(Equal(createdConn.Name))
		})
	})

	Context("Refresh", func() {
		It("updates expiresAt for an external connection", func() {
			apiKey := "rre_unit_test_key"
			secret := newTestSecret(apiKey, []string{"default"})

			conn := &tfv1.TensorFusionConnection{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "ext-1234",
					Namespace: "default",
					Labels: map[string]string{
						constants.ExternalConnectionLabelKey: constants.TrueStringValue,
					},
				},
				Spec: tfv1.TensorFusionConnectionSpec{
					WorkloadName: "wl",
					ExternalClient: &tfv1.ExternalClientSpec{
						ClientID:   "c1",
						TTLSeconds: 3600,
					},
				},
				Status: tfv1.TensorFusionConnectionStatus{
					ExpiresAt: ptrTo(metav1.NewTime(time.Now().Add(-time.Minute))),
				},
			}

			k8s = fake.NewClientBuilder().
				WithScheme(scheme).
				WithObjects(secret, conn).
				WithStatusSubresource(&tfv1.TensorFusionConnection{}).
				Build()
			r = NewExternalConnectionRouter(k8s)

			ctx, w := newParamCtx(http.MethodPost, "/api/v1/external-connections/default/ext-1234/refresh", apiKey,
				gin.Params{{Key: "namespace", Value: "default"}, {Key: "name", Value: "ext-1234"}})

			r.Refresh(ctx)
			Expect(w.Code).To(Equal(http.StatusOK))

			updated := &tfv1.TensorFusionConnection{}
			Expect(k8s.Get(context.Background(), types.NamespacedName{Name: "ext-1234", Namespace: "default"}, updated)).To(Succeed())
			Expect(updated.Status.ExpiresAt).ToNot(BeNil())
			Expect(updated.Status.ExpiresAt.Time).To(BeTemporally(">", time.Now().Add(30*time.Minute)))
		})
	})
})

func ptrTo[T any](v T) *T { return &v }

