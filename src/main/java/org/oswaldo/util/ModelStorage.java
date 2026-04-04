package org.oswaldo.util;

import org.oswaldo.model.NeuralNetwork;

import java.io.*;

public class ModelStorage {
    public static void saveModel(NeuralNetwork model, String path) throws IOException {
        try (ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(path))) {
            out.writeObject(model);
        }
    }

    public static NeuralNetwork loadModel(String path) throws IOException, ClassNotFoundException {
        try (ObjectInputStream in = new ObjectInputStream(new FileInputStream(path))) {
            return (NeuralNetwork) in.readObject();
        }
    }
}