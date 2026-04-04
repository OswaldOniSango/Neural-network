package org.oswaldo;

import java.io.File;

import static org.oswaldo.util.TrainModel.loadAndPredict;
import static org.oswaldo.util.TrainModel.trainAndSaveModel;

public class NeuralNetworkSumApp {
    public static void main(String[] args) {
        String modelPath = "src/main/resources/model/sum-model.ser";

        try {
            File modelFile = new File(modelPath);

            if (!modelFile.exists()) {
                System.out.println("No existe modelo guardado. Se va a entrenar uno nuevo...");
                trainAndSaveModel(modelPath);
            } else {
                System.out.println("Ya existe un modelo guardado. No se reentrena.");
            }

            loadAndPredict(modelPath);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
