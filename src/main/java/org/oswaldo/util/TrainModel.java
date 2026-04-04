package org.oswaldo.util;

import org.oswaldo.model.NeuralNetwork;

import java.io.IOException;

public class TrainModel {
    public static void trainAndSaveModel(String modelPath) throws IOException {
        double[][] trainingInputs = {
                {0.0, 0.0},
                {0.0, 0.1},
                {0.1, 0.2},
                {0.2, 0.3},
                {0.3, 0.4},
                {0.4, 0.5},
                {0.5, 0.6},
                {0.6, 0.7},
                {0.7, 0.8},
                {0.8, 0.9},
                {0.9, 1.0},
                {1.0, 1.0}
        };

        double[][] trainingOutputs = {
                {0.0},
                {0.1},
                {0.3},
                {0.5},
                {0.7},
                {0.9},
                {1.1},
                {1.3},
                {1.5},
                {1.7},
                {1.9},
                {2.0}
        };

        NeuralNetwork nn = new NeuralNetwork(2, 6, 1);
        nn.train(trainingInputs, trainingOutputs, 20000, 0.1);

        ModelStorage.saveModel(nn, modelPath);
        System.out.println("\nModelo entrenado y guardado en: " + modelPath);
    }

    public static void loadAndPredict(String modelPath) throws IOException, ClassNotFoundException {
        NeuralNetwork model = ModelStorage.loadModel(modelPath);
        model.printModel();
        System.out.println("\nModelo cargado desde: " + modelPath);

        double[][] tests = {
                {0.1, 0.1},
                {0.2, 0.5},
                {0.3, 0.7},
                {0.8, 0.4},
                {1.0, 0.9}
        };

        System.out.println("\nPruebas:");
        for (double[] test : tests) {
            double[] prediction = model.predict(test);
            System.out.printf("%.2f + %.2f = %.2f%n", test[0], test[1], prediction[0]);
        }
    }
}
