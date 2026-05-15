package org.oswaldo.model;

import java.io.Serializable;
import java.util.Random;

/**
 * Implementación de una red neuronal artificial con una capa oculta.
 * Utiliza la función de activación sigmoide en la capa oculta y
 * una salida lineal en la capa de salida.
 *
 * <p>Arquitectura: inputSize → hiddenSize → outputSize</p>
 *
 * <p>Soporta serialización para guardar y cargar el modelo entrenado.</p>
 */
public class NeuralNetwork implements Serializable {
    private static final long serialVersionUID = 1L;

    /** Número de neuronas en la capa de entrada. */
    private final int inputSize;

    /** Número de neuronas en la capa oculta. */
    private final int hiddenSize;

    /** Número de neuronas en la capa de salida. */
    private final int outputSize;

    /** Pesos entre la capa de entrada y la capa oculta [inputSize][hiddenSize]. */
    private final double[][] weightsInputHidden;

    /** Sesgos (bias) de la capa oculta [hiddenSize]. */
    private final double[] biasHidden;

    /** Pesos entre la capa oculta y la capa de salida [hiddenSize][outputSize]. */
    private final double[][] weightsHiddenOutput;

    /** Sesgos (bias) de la capa de salida [outputSize]. */
    private final double[] biasOutput;

    /** Generador de números aleatorios para inicializar los pesos. Marcado transient para serialización. */
    private transient Random random = new Random();

    /**
     * Crea una nueva red neuronal con los tamaños de capas especificados
     * e inicializa los pesos aleatoriamente en el rango [-1, 1].
     *
     * @param inputSize  Número de neuronas en la capa de entrada.
     * @param hiddenSize Número de neuronas en la capa oculta.
     * @param outputSize Número de neuronas en la capa de salida.
     */
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

    /**
     * Inicializa todos los pesos y sesgos de la red con valores aleatorios
     * uniformes en el rango [-1, 1].
     */
    private void initializeWeights() {
        // Inicializar pesos entrada → oculta
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < hiddenSize; j++) {
                weightsInputHidden[i][j] = random.nextDouble() * 2 - 1;
            }
        }

        // Inicializar sesgos de la capa oculta
        for (int j = 0; j < hiddenSize; j++) {
            biasHidden[j] = random.nextDouble() * 2 - 1;
        }

        // Inicializar pesos oculta → salida
        for (int j = 0; j < hiddenSize; j++) {
            for (int k = 0; k < outputSize; k++) {
                weightsHiddenOutput[j][k] = random.nextDouble() * 2 - 1;
            }
        }

        // Inicializar sesgos de la capa de salida
        for (int k = 0; k < outputSize; k++) {
            biasOutput[k] = random.nextDouble() * 2 - 1;
        }
    }

    /**
     * Método invocado automáticamente después de deserializar el objeto.
     * Reinicializa el generador de números aleatorios (campo transient).
     *
     * @return La misma instancia restaurada.
     */
    private Object readResolve() {
        this.random = new Random();
        return this;
    }

    /**
     * Función de activación sigmoide: f(x) = 1 / (1 + e^(-x)).
     * Comprime cualquier valor real al intervalo (0, 1).
     *
     * @param x Valor de entrada.
     * @return Valor de salida entre 0 y 1.
     */
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /**
     * Derivada de la función sigmoide expresada en términos de su salida:
     * f'(output) = output * (1 - output).
     * Usada durante la retropropagación para calcular gradientes.
     *
     * @param output Valor de salida de la neurona (ya activado).
     * @return Gradiente local de la función sigmoide.
     */
    private double sigmoidDerivative(double output) {
        return output * (1.0 - output);
    }

    /**
     * Realiza una pasada hacia adelante (forward pass) y devuelve las predicciones
     * de la red para el vector de entrada dado.
     *
     * <p>La capa oculta utiliza activación sigmoide; la capa de salida es lineal.</p>
     *
     * @param inputs Vector de entrada de tamaño {@code inputSize}.
     * @return Vector de salida de tamaño {@code outputSize}.
     */
    public double[] predict(double[] inputs) {
        double[] hidden = new double[hiddenSize];
        double[] outputs = new double[outputSize];

        // Calcular activaciones de la capa oculta con función sigmoide
        for (int j = 0; j < hiddenSize; j++) {
            double sum = biasHidden[j];
            for (int i = 0; i < inputSize; i++) {
                sum += inputs[i] * weightsInputHidden[i][j];
            }
            hidden[j] = sigmoid(sum);
        }

        // Calcular salidas de la capa de salida (activación lineal)
        for (int k = 0; k < outputSize; k++) {
            double sum = biasOutput[k];
            for (int j = 0; j < hiddenSize; j++) {
                sum += hidden[j] * weightsHiddenOutput[j][k];
            }
            outputs[k] = sum; // salida lineal
        }

        return outputs;
    }

    /**
     * Entrena la red neuronal usando el algoritmo de retropropagación (backpropagation)
     * con descenso de gradiente estocástico (SGD) muestra a muestra.
     *
     * @param trainingInputs  Matriz de entradas de entrenamiento [muestras][inputSize].
     * @param trainingOutputs Matriz de salidas esperadas [muestras][outputSize].
     * @param epochs          Número de épocas (iteraciones completas sobre el dataset).
     * @param learningRate    Tasa de aprendizaje que controla el tamaño del paso del gradiente.
     */
    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs, double learningRate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double totalError = 0.0;

            // Iterar sobre cada muestra de entrenamiento
            for (int sample = 0; sample < trainingInputs.length; sample++) {
                double[] inputs = trainingInputs[sample];
                double[] expected = trainingOutputs[sample];

                double[] hidden = new double[hiddenSize];
                double[] outputs = new double[outputSize];

                // --- Forward pass: entrada → capa oculta ---
                for (int j = 0; j < hiddenSize; j++) {
                    double sum = biasHidden[j];
                    for (int i = 0; i < inputSize; i++) {
                        sum += inputs[i] * weightsInputHidden[i][j];
                    }
                    hidden[j] = sigmoid(sum);
                }

                // --- Forward pass: capa oculta → salida (lineal) ---
                for (int k = 0; k < outputSize; k++) {
                    double sum = biasOutput[k];
                    for (int j = 0; j < hiddenSize; j++) {
                        sum += hidden[j] * weightsHiddenOutput[j][k];
                    }
                    outputs[k] = sum;
                }

                // --- Calcular error en la capa de salida (diferencia cuadrática) ---
                double[] outputErrors = new double[outputSize];
                for (int k = 0; k < outputSize; k++) {
                    outputErrors[k] = expected[k] - outputs[k];
                    totalError += outputErrors[k] * outputErrors[k];
                }

                // --- Retropropagar el error hacia la capa oculta ---
                double[] hiddenErrors = new double[hiddenSize];
                for (int j = 0; j < hiddenSize; j++) {
                    double error = 0.0;
                    for (int k = 0; k < outputSize; k++) {
                        error += outputErrors[k] * weightsHiddenOutput[j][k];
                    }
                    // Aplicar la derivada de sigmoide para obtener el delta de la capa oculta
                    hiddenErrors[j] = error * sigmoidDerivative(hidden[j]);
                }

                // --- Actualizar pesos oculta → salida ---
                for (int j = 0; j < hiddenSize; j++) {
                    for (int k = 0; k < outputSize; k++) {
                        weightsHiddenOutput[j][k] += learningRate * outputErrors[k] * hidden[j];
                    }
                }

                // --- Actualizar sesgos de la capa de salida ---
                for (int k = 0; k < outputSize; k++) {
                    biasOutput[k] += learningRate * outputErrors[k];
                }

                // --- Actualizar pesos entrada → oculta ---
                for (int i = 0; i < inputSize; i++) {
                    for (int j = 0; j < hiddenSize; j++) {
                        weightsInputHidden[i][j] += learningRate * hiddenErrors[j] * inputs[i];
                    }
                }

                // --- Actualizar sesgos de la capa oculta ---
                for (int j = 0; j < hiddenSize; j++) {
                    biasHidden[j] += learningRate * hiddenErrors[j];
                }
            }

            // Imprimir el error total cada 1000 épocas para monitorear el entrenamiento
            if (epoch % 1000 == 0) {
                System.out.printf("Epoch %d - Error total: %.8f%n", epoch, totalError);
            }
        }
    }

    /**
     * Imprime en consola todos los pesos y sesgos actuales de la red,
     * útil para depuración e inspección del modelo.
     */
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
