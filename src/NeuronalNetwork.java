import java.util.ArrayList;
import java.util.List;

public class NeuronalNetwork {

    private NetworkLayer inputLayer;
    private NetworkLayer outputLayer;
    private List<NetworkLayer> hiddenLayers = new ArrayList<>();

    private double maxNetworkError = 0.005;

    public NeuronalNetwork(int[] nodeCounts, int hiddenLayersCount){
        if(nodeCounts.length != 3){
            System.err.println("Invalid number of node counts! Should be exactly 3 (input, hidden, output)!");
            System.exit(0);
        }
        this.inputLayer = new NetworkLayer(nodeCounts[0]);
        this.outputLayer = new NetworkLayer(nodeCounts[2]);

        NetworkLayer last = this.inputLayer;
        for(int i = 0; i < hiddenLayersCount; i++){
            NetworkLayer current = new NetworkLayer(nodeCounts[1]);
            last.setChildLayer(current);
            current.setParentLayer(last);
            this.hiddenLayers.add(current);
            last = current;
        }
        last.setChildLayer(this.outputLayer);
        this.outputLayer.setParentLayer(last);

        this.inputLayer.initArrays();
        this.outputLayer.initArrays();
        for(int i = 0; i < hiddenLayersCount; i++){
            this.hiddenLayers.get(i).initArrays();
        }
    }

    public void setMaxNetworkError(double maxNetworkError) {
        this.maxNetworkError = maxNetworkError;
    }

    public void setLearningRate(double learningRate) {
        this.inputLayer.setLearningRate(learningRate);
        for (NetworkLayer hiddenLayer : hiddenLayers) {
            hiddenLayer.setLearningRate(learningRate);
        }
        this.outputLayer.setLearningRate(learningRate);
    }

    public void setUseMomentum(boolean useMomentum) {
        this.inputLayer.setUseMomentum(useMomentum);
        for (NetworkLayer hiddenLayer : hiddenLayers) {
            hiddenLayer.setUseMomentum(useMomentum);
        }
        this.outputLayer.setUseMomentum(useMomentum);
    }

    public void setMomentumFactor(double momentumFactor) {
        this.inputLayer.setMomentumFactor(momentumFactor);
        for (NetworkLayer hiddenLayer : hiddenLayers) {
            hiddenLayer.setMomentumFactor(momentumFactor);
        }
        this.outputLayer.setMomentumFactor(momentumFactor);
    }

    public void setLinearOutput(boolean linearOutput) {
        this.inputLayer.setLinearOutput(linearOutput);
        for (NetworkLayer hiddenLayer : hiddenLayers) {
            hiddenLayer.setLinearOutput(linearOutput);
        }
        this.outputLayer.setLinearOutput(linearOutput);
    }

    public void learn(List<DigitImage> images){
        double error = 1.0;
        double sumError = 0.0;
        int round = 1;
        while(error > maxNetworkError){
            for (DigitImage image : images) {
                // Set input
                this.inputLayer.setNeuronValues(image.getData());
                // Set output
                double[] desiredOutput = new double[this.outputLayer.getNumberOfNodes()];
                for (int j = 0; j < this.outputLayer.getNumberOfNodes(); j++) {
                    desiredOutput[j] = 0.0;
                }
                desiredOutput[image.getLabel()] = 1.0;
                this.outputLayer.setDesiredValues(desiredOutput);

                feedForward();
                backPropagate();

                // Calculate network error
                sumError += calculateError();
            }
            error = sumError/(double)images.size();
            sumError = 0.0;
            System.out.println("Round "+round+", Network Error: "+error);
            round++;
        }
    }

    public int classify(DigitImage image){
        this.inputLayer.setNeuronValues(image.getData());
        feedForward();
        double highestValue = 0;
        int highestIndex = -1;
        double[] outputValues = this.outputLayer.getNeuronValues();
        for(int i = 0; i < this.outputLayer.getNumberOfNodes(); i++){
            if(outputValues[i] > highestValue){
                highestIndex = i;
                highestValue = outputValues[i];
            }
        }
        return highestIndex;
    }

    private void feedForward() {
        inputLayer.calculateNeuronValues();
        for (NetworkLayer hiddenLayer : hiddenLayers) {
            hiddenLayer.calculateNeuronValues();
        }
        outputLayer.calculateNeuronValues();
    }

    private void backPropagate() {
        outputLayer.calculateErrors();
        for(int i = hiddenLayers.size()-1; i >= 0; i--) {
            hiddenLayers.get(i).calculateErrors();
        }
        for(int i = hiddenLayers.size()-1; i >= 0; i--){
            hiddenLayers.get(i).adjustWeights();
        }
        inputLayer.adjustWeights();
    }

    private double calculateError() {
        double error = 0.0;
        for (int i = 0; i < outputLayer.getNumberOfNodes(); i++) {
            error += Math.pow(outputLayer.getNeuronValues()[i] - outputLayer.getDesiredValues()[i], 2);
        }
        error = error / outputLayer.getNumberOfNodes();
        return error;
    }
}
