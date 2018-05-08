import java.io.IOException;
import java.util.List;

public class Main {

    private static List<DigitImage> images;
    private static List<DigitImage> testImages;

    public static void loadImageData() {
        DigitImageLoadingService dils = new DigitImageLoadingService(
                "data\\train-labels.idx1-ubyte",
                "data\\train-images.idx3-ubyte");
        DigitImageLoadingService dilsTest = new DigitImageLoadingService(
                "data\\t10k-labels.idx1-ubyte",
                "data\\t10k-images.idx3-ubyte");
        try {
            images = dils.loadDigitImages();
            testImages = dilsTest.loadDigitImages();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public static void main(String[] args) {
        loadImageData();

        NeuronalNetwork nn = new NeuronalNetwork(new int[]{784, 89, 10}, 3);
        nn.setLinearOutput(false);
        nn.setMaxNetworkError(0.005);
        nn.setLearningRate(0.2);
        nn.setUseMomentum(true);
        nn.setMomentumFactor(0.9);


        nn.learn(images);

        ConfusionMatrix confusionMatrix = new ConfusionMatrix();

        for (DigitImage testImage : testImages) {
            confusionMatrix.addEntry(testImage.getLabel(), nn.classify(testImage));
        }

        System.out.println();
        confusionMatrix.printToConsole();
    }
}
