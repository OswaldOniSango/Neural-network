package org.oswaldo.model;

import java.io.Serializable;
import java.util.Random;

public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;

    private final double[][] weightsInputHidden;
    private final double[] biasHidden;

    private final double[][] weightsHiddenOutput;
    private final double[] biasOutput;

    private transient Random random = new Random();

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.weightsInputHidden = new double[inputSize][hiddenSize];
        this.biasHidden = new double[hiddenSize];

        this.weightsHiddenOutput = new double[hiddenSize][outputSize];
        this.biasOutput = new double[outputSize];

        initializeWeights();
    }

    private void initializeWeights() {
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = random.nextDouble() * 2 - 1;
            }
        }

        for (int j = 0; j < hiddenSize; j++) {
            biasHidden[j] = random.nextDouble() * 2 - 1;
        }

        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                weightsHiddenOutput[j][k] = random.nextDouble() * 2 - 1;
            }
        }

        for (int k = 0; k < outputSize; k++) {
            biasOutput[k] = random.nextDouble() * 2 - 1;
        }
    }

    private Object readResolve() {
        this.random = new Random();
        return this;
    }

    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    private double sigmoidDerivative(double output) {
        return output * (1.0 - output);
    }

    public double[] predict(double[] inputs) {
        double[] hidden = new double[hiddenSize];
        double[] outputs = new double[outputSize];

        for (int j = 0; j < hiddenSize; j++) {
            double sum = biasHidden[j];
            for (int i = 0; i < inputSize; i++) {
                sum += inputs[i] * weightsInputHidden[i][j];
            }
            hidden[j] = sigmoid(sum);
        }

        for (int k = 0; k < outputSize; k++) {
            double sum = biasOutput[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * weightsHiddenOutput[j][k];
            }
            outputs[k] = sum; // salida lineal
        }

        return outputs;
    }
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;

            for (int sample = 0; sample < trainingInputs.length; sample++) {
                double[] inputs = trainingInputs[sample];
                double[] expected = trainingOutputs[sample];

                double[] hidden = new double[hiddenSize];
                double[] outputs = new double[outputSize];

                // FORWARD PASS
                for (int j = 0; j < hiddenSize; j++) {
                    double sum = biasHidden[j];
                    for (int i = 0; i < inputSize; i++) {
                        sum += inputs[i] * weightsInputHidden[i][j];
                    }
                    hidden[j] = sigmoid(sum);
                }

                for (int k = 0; k < outputSize; k++) {
                    double sum = biasOutput[k];
                    for (int j = 0; j < hiddenSize; j++) {
                        sum += hidden[j] * weightsHiddenOutput[j][k];
                    }
                    outputs[k] = sum; // salida lineal
                }

                // ERROR EN LA SALIDA
                double[] outputErrors = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    outputErrors[k] = expected[k] - outputs[k];
                    totalError += outputErrors[k] * outputErrors[k];
                }

                // ERROR EN LA CAPA OCULTA
                double[] hiddenErrors = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    double error = 0.0;
                    for (int k = 0; k < outputSize; k++) {
                        error += outputErrors[k] * weightsHiddenOutput[j][k];
                    }
                    hiddenErrors[j] = error * sigmoidDerivative(hidden[j]);
                }

                // ACTUALIZAR PESOS HIDDEN -> OUTPUT
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputErrors[k] * hidden[j];
                    }
                }

                // ACTUALIZAR BIAS OUTPUT
                for (int k = 0; k < outputSize; k++) {
                    biasOutput[k] += learningRate * outputErrors[k];
                }

                // ACTUALIZAR PESOS INPUT HIDDEN
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * inputs[i];
                    }
                }

                // ACTUALIZAR BIAS HIDDEN
                for (int j = 0; j < hiddenSize; j++) {
                    biasHidden[j] += learningRate * hiddenErrors[j];
                }
            }

            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error total: %.8f%n", epoch, totalError);
            }
        }
    }

    public void printModel() {
        System.out.println("weightsInputHidden:");
        for (int i = 0; i < weightsInputHidden.length; i++) {
            for (int j = 0; j < weightsInputHidden[i].length; j++) {
                System.out.printf("weightsInputHidden[%d][%d] = %.10f%n", i, j, weightsInputHidden[i][j]);
            }
        }

        System.out.println("\nbiasHidden:");
        for (int i = 0; i < biasHidden.length; i++) {
            System.out.printf("biasHidden[%d] = %.10f%n", i, biasHidden[i]);
        }

        System.out.println("\nweightsHiddenOutput:");
        for (int i = 0; i < weightsHiddenOutput.length; i++) {
            for (int j = 0; j < weightsHiddenOutput[i].length; j++) {
                System.out.printf("weightsHiddenOutput[%d][%d] = %.10f%n", i, j, weightsHiddenOutput[i][j]);
            }
        }

        System.out.println("\nbiasOutput:");
        for (int i = 0; i < biasOutput.length; i++) {
            System.out.printf("biasOutput[%d] = %.10f%n", i, biasOutput[i]);
        }
    }
}