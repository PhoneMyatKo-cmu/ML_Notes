#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float train[][2] = {{0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 8}};

#define train_count (sizeof(train) / sizeof(train[0]))

float rand_float(void) { return (float)rand() / (float)RAND_MAX; }

float cost(float w, float b) {
  float mse = 0.0f;
  for (size_t i = 0; i < train_count; i++) {
    float x = train[i][0];
    float y = train[i][1];
    float y_hat = x * w + b;
    float d = y - y_hat;
    mse += d * d;
  }
  return mse /= train_count;
}

float model(float x, float w, float b) { return x * w + b; }

int main() {
  srand(time(0));
  float w = rand_float() * 10.0f;
  float b = rand_float() * 5.0f;
  float epsilon = 1e-3;
  float alpha = 1e-3;
  for (size_t i = 0; i < 500; i++) {
    float dcost = (cost(w + epsilon, b) - cost(w, b)) / epsilon;
    float db = (cost(w, b + epsilon) - cost(w, b)) / epsilon;

    w -= alpha * dcost;
    b -= alpha * db;
    printf("MSE:%f\n", cost(w, b));
  }
  printf("-------------------------\n");
  printf("Weight:%f,Bias:%f\n", w, b);
  printf("Model Prediction of %f:%f\n", 4.0, model(4, w, b));
  return 0;
}
