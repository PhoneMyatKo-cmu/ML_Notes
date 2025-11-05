#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][3] = {
    {0, 0, 0},
    {0, 1, 1},
    {1, 0, 1},
    {1, 1, 1},
};

float sigmoidf(float y) { return 1.f / (1.f + expf(-y)); }

#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float cost(float w1, float w2) {
  float mse = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x1 = train[i][0];
    float x2 = train[i][1];
    float y = train[i][2];
    float y_hat = sigmoidf(x1 * w1 + x2 * w2);
    float d = y - y_hat;
    mse += d * d;
  }
  return mse /= train_count;
}

float model(float x1, float x2, float w1, float w2) {

  return sigmoidf(x1 * w1 + x2 * w2);
}

int main() {
  srand(69);
  float w1 = rand_float();
  float w2 = rand_float();
  float epsilon = 1e-3;
  float alpha = 1e-2;
  for (size_t i = 0; i < 4000; i++) {
    float dw1 = (cost(w1 + epsilon, w2) - cost(w1, w2)) / epsilon;
    float dw2 = (cost(w1, w2 + epsilon) - cost(w1, w2)) / epsilon;

    w1 -= alpha * dw1;
    w2 -= alpha * dw2;
    printf("MSE:%f,w1:%f,w2:%f\n", cost(w1, w2), w1, w2);
  }
  printf("-------------------------\n");
  printf("w1:%f,w2:%f\n", w1, w2);
  for (int i = 0; i < train_count; i++) {
    printf("Model Prediction of %f and %f:%f\n", train[i][0], train[i][1],
           model(train[i][0], train[i][1], w1, w2));
  }
  return 0;
}
