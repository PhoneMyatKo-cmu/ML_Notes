public class xor_two_layers {

  public static void main(String[] args) {

    // first construct training data

    float[][] trainingData = { { 0, 0, 0 }, { 1, 0, 1 }, { 0, 1, 1 }, { 1, 1, 0 } };
    Model model = new Model();
    model.train(1000000, 0.01f, trainingData, 0.01f);
    System.out.println("--------Prediction---------");
    model.predict(trainingData);
  }
}

class Model {

  float or_w1;
  float or_w2;
  float or_b;

  float nand_w1;
  float nand_w2;
  float nand_b;

  float and_w1;
  float and_w2;
  float and_b;

  public Model() {
    this.or_w1 = (float) Math.random();
    this.or_w2 = (float) Math.random();
    this.or_b = (float) Math.random();
    this.nand_w1 = (float) Math.random();
    this.nand_w2 = (float) Math.random();
    this.nand_b = (float) Math.random();
    this.and_w1 = (float) Math.random();
    this.and_w2 = (float) Math.random();
    this.and_b = (float) Math.random();
    System.out.println(this);
  }

  @Override
  public String toString() {
    return "or_w1:" + or_w1 + ",or_w2:" + or_w2 + ",or_b:" + or_b +
        "\nnand_w1:" + nand_w1 + ",nand_w2:" + nand_w2 + ",nand_b:" + nand_b +
        "\nand_w1:" + and_w1 + ",and_w2:" + and_w2 + ",and_b:" + and_b;
  }

  public float cost(float trainingData[][]) {
    float mse = 0;
    for (int i = 0; i < trainingData.length; i++) {
      float x1 = trainingData[i][0];
      float x2 = trainingData[i][1];
      float node1 = sigmoid(or_w1 * x1 + or_w2 * x2 + or_b);
      float node2 = sigmoid(nand_w1 * x1 + nand_w2 * x2 + nand_b);
      float y_hat = sigmoid(and_w1 * node1 + and_w2 * node2 + and_b);
      float y = trainingData[i][2];
      mse += (y - y_hat) * (y - y_hat);
    }
    return mse / trainingData.length;
  }

  public void train(int epoch, float learning_rate, float trainingData[][],
      float eps) {
    for (int i = 0; i < epoch; i++) {
      float c = cost(trainingData);

      float saved = this.or_w1;
      this.or_w1 += eps;
      float dOrW1 = (cost(trainingData) - c) / eps;
      this.or_w1 = saved;

      saved = this.or_w2;
      this.or_w2 += eps;
      float dOrW2 = (cost(trainingData) - c) / eps;
      this.or_w2 = saved;

      saved = this.or_b;
      this.or_b += eps;
      float dOrB = (cost(trainingData) - c) / eps;
      this.or_b = saved;

      saved = this.nand_w1;
      this.nand_w1 += eps;
      float dNandW1 = (cost(trainingData) - c) / eps;
      this.nand_w1 = saved;

      saved = this.nand_w2;
      this.nand_w2 += eps;
      float dNandW2 = (cost(trainingData) - c) / eps;
      this.nand_w2 = saved;

      saved = this.nand_b;
      this.nand_b += eps;
      float dNandB = (cost(trainingData) - c) / eps;
      this.nand_b = saved;

      saved = this.and_w1;
      this.and_w1 += eps;
      float dAndW1 = (cost(trainingData) - c) / eps;
      this.and_w1 = saved;

      saved = this.and_w2;
      this.and_w2 += eps;
      float dAndW2 = (cost(trainingData) - c) / eps;
      this.and_w2 = saved;

      saved = this.and_b;
      this.and_b += eps;
      float dAndB = (cost(trainingData) - c) / eps;
      this.and_b = saved;

      // change weights and bias
      this.or_w1 -= learning_rate * dOrW1;
      this.or_w2 -= learning_rate * dOrW2;
      this.or_b -= learning_rate * dOrB;
      this.nand_w1 -= learning_rate * dNandW1;
      this.nand_w2 -= learning_rate * dNandW2;
      this.nand_b -= learning_rate * dNandB;
      this.and_w1 -= learning_rate * dAndW1;
      this.and_w2 -= learning_rate * dAndW2;
      this.and_b -= learning_rate * dAndB;

      System.out.println("Epoch " + (i + 1) + " ,MSE=" + cost(trainingData));
    }
  }

  public float sigmoid(float x) {
    return (float) (1 / (1 + Math.exp(-x)));
  }

  public int predict(float x1, float x2) {
    float node1 = sigmoid(or_w1 * x1 + or_w2 * x2 + or_b);
    float node2 = sigmoid(nand_w1 * x1 + nand_w2 * x2 + nand_b);
    float y_hat = sigmoid(and_w1 * node1 + and_w2 * node2 + and_b);
    return y_hat > 0.2 ? 1 : 0;
  }

  public void predict(float testData[][]) {
    for (int i = 0; i < testData.length; i++) {
      float x1 = testData[i][0];
      float x2 = testData[i][1];
      System.out.println(x1 + " ^ " + x2 + " = " + predict(x1, x2));
    }
  }
}
