#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][3] = {
    {0, 0, 0},
    {0, 1, 0},
    {1, 0, 0},
    {1, 1, 1},
};

#define train_count (sizeof(train) / sizeof(train[0]))

float sigmoidf(float y) { return 1.f / (1.f + expf(-y)); }

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float cost(float w1, float w2, float bias) {
  float mse = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float y_hat = sigmoidf(x1 * w1 + x2 * w2 + bias);
    float d = y - y_hat;
    mse += d * d;
  }
  return mse /= train_count;
}

float model(float x1, float x2, float w1, float w2, float bias) {

  return sigmoidf(x1 * w1 + x2 * w2 + bias);
}

int main() {
  srand(time(0));
  float w1 = rand_float();
  float w2 = rand_float();
  float bias = rand_float();
  float epsilon = 1e-3;
  float alpha = 1e-2;
  for (size_t i = 0; i < 10000 * 100; i++) {
    float dw1 = (cost(w1 + epsilon, w2, bias) - cost(w1, w2, bias)) / epsilon;
    float dw2 = (cost(w1, w2 + epsilon, bias) - cost(w1, w2, bias)) / epsilon;
    float dbias = (cost(w1, w2, bias + epsilon) - cost(w1, w2, bias)) / epsilon;
    w1 -= alpha * dw1;
    w2 -= alpha * dw2;
    bias -= alpha * dbias;
    printf("MSE:%f,w1:%f,w2:%f,bias:%f\n", cost(w1, w2, bias), w1, w2, bias);
  }
  printf("-------------------------\n");
  printf("w1:%f,w2:%f,bias:%f\n", w1, w2, bias);
  for (int i = 0; i < train_count; i++) {
    printf("Model Prediction of %.1f | %.1f:%.1f\n", train[i][0], train[i][1],
           model(train[i][0], train[i][1], w1, w2, bias));
  }
  return 0;
}
