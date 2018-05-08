import java.util.List;

public class NeuronalNetwork {

    private NetworkLayer inputLayer;
    private NetworkLayer hiddenLayer;
    private NetworkLayer outputLayer;

    private double maxNetworkError = 0.005;

    public NeuronalNetwork(int[] nodeCounts){
        if(nodeCounts.length != 3){
            System.err.println("Invalid number of node counts! Should be exactly 3 (input, hidden, output)!");
            System.exit(0);
        }
        this.inputLayer = new NetworkLayer(nodeCounts[0]);
        this.hiddenLayer = new NetworkLayer(nodeCounts[1]);
        this.outputLayer = new NetworkLayer(nodeCounts[2]);

        this.inputLayer.setChildLayer(this.hiddenLayer);
        this.hiddenLayer.setParentLayer(this.inputLayer);
        this.hiddenLayer.setChildLayer(this.outputLayer);
        this.outputLayer.setParentLayer(this.hiddenLayer);

        this.inputLayer.initArrays();
        this.hiddenLayer.initArrays();
        this.outputLayer.initArrays();
    }

    public void setMaxNetworkError(double maxNetworkError) {
        this.maxNetworkError = maxNetworkError;
    }

    public void setLearningRate(double learningRate) {
        this.inputLayer.setLearningRate(learningRate);
        this.hiddenLayer.setLearningRate(learningRate);
        this.outputLayer.setLearningRate(learningRate);
    }

    public void setUseMomentum(boolean useMomentum) {
        this.inputLayer.setUseMomentum(useMomentum);
        this.hiddenLayer.setUseMomentum(useMomentum);
        this.outputLayer.setUseMomentum(useMomentum);
    }

    public void setMomentumFactor(double momentumFactor) {
        this.inputLayer.setMomentumFactor(momentumFactor);
        this.hiddenLayer.setMomentumFactor(momentumFactor);
        this.outputLayer.setMomentumFactor(momentumFactor);
    }

    public void setLinearOutput(boolean linearOutput) {
        this.inputLayer.setLinearOutput(linearOutput);
        this.hiddenLayer.setLinearOutput(linearOutput);
        this.outputLayer.setLinearOutput(linearOutput);
    }

    public void learn(List<DigitImage> images){
        double error = 1.0;
        double sumError = 0.0;
        int round = 1;
        while(error > maxNetworkError){
            for(int i = 0; i < images.size(); i++) {
                // Set input
                this.inputLayer.setNeuronValues(images.get(i).getData());
                // Set output
                double[] desiredOutput = new double[this.outputLayer.getNumberOfNodes()];
                for(int j = 0; j < this.outputLayer.getNumberOfNodes(); j++){
                    desiredOutput[j] = 0.0;
                }
                desiredOutput[images.get(i).getLabel()] = 1.0;
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

    public void feedForward() {
        inputLayer.calculateNeuronValues();
        hiddenLayer.calculateNeuronValues();
        outputLayer.calculateNeuronValues();
    }

    public void backPropagate() {
        outputLayer.calculateErrors();
        hiddenLayer.calculateErrors();
        hiddenLayer.adjustWeights();
        inputLayer.adjustWeights();
    }

    public double calculateError() {
        double error = 0.0;
        for (int i = 0; i < outputLayer.getNumberOfNodes(); i++) {
            error += Math.pow(outputLayer.getNeuronValues()[i] - outputLayer.getDesiredValues()[i], 2);
        }
        error = error / outputLayer.getNumberOfNodes();
        return error;
    }
}
