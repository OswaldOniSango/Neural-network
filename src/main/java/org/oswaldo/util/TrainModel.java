package org.oswaldo.util;

import org.oswaldo.model.NeuralNetwork;

import java.io.IOException;
import java.util.Random;

public class TrainModel {
    public static void trainAndSaveModel(String modelPath) throws IOException {
        // Generamos un dataset aleatorio que cubra todo el cuadrado [0,1] x [0,1]
        // (con seed fija para reproducibilidad)
        int n = 500;
        double[][] trainingInputs = new double[n][2];
        double[][] trainingOutputs = new double[n][1];
        Random rnd = new Random(42);
        for (int i = 0; i < n; i++) {
            double a = rnd.nextDouble();
            double b = rnd.nextDouble();
            trainingInputs[i][0] = a;
            trainingInputs[i][1] = b;
            trainingOutputs[i][0] = a + b;
        }

        NeuralNetwork nn = new NeuralNetwork(2, 4, 1);
        nn.train(trainingInputs, trainingOutputs, 5000, 0.05);

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
