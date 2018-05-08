public class NetworkLayer {

    private boolean useMomentum = true;
    private double learningRate = 0.2;
    private double momentumFactor = 0.9;
    private boolean linearOutput = false;

    private NetworkLayer childLayer = null;
    private NetworkLayer parentLayer = null;

    private int numberOfNodes = 0;
    private double[] neuronValues;
    private double[][] weights;
    private double[][] weightChanges;
    private double[] errors;
    private double[] biasWeights;
    private double[] biasValues;
    private double[] desiredValues;

    public NetworkLayer(int numberOfNodes){
        this.numberOfNodes = numberOfNodes;
    }

    public void setChildLayer(NetworkLayer childLayer) {
        this.childLayer = childLayer;
    }

    public void setParentLayer(NetworkLayer parentLayer) {
        this.parentLayer = parentLayer;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public void setUseMomentum(boolean useMomentum) {
        this.useMomentum = useMomentum;
    }

    public void setMomentumFactor(double momentumFactor) {
        this.momentumFactor = momentumFactor;
    }

    public void setLinearOutput(boolean linearOutput) {
        this.linearOutput = linearOutput;
    }

    public void setDesiredValues(double[] desiredValues) {
        this.desiredValues = desiredValues;
    }

    public void setNeuronValues(double[] neuronValues) {
        this.neuronValues = neuronValues;
    }

    public int getNumberOfNodes() {
        return numberOfNodes;
    }

    public double[] getNeuronValues() {
        return neuronValues;
    }

    public double[] getDesiredValues() {
        return desiredValues;
    }

    public void initArrays(){
        this.neuronValues = new double[this.numberOfNodes];
        this.errors = new double[this.numberOfNodes];
        this.desiredValues = new double[this.numberOfNodes];

        if(childLayer != null) {
            this.weights = new double[this.numberOfNodes][this.childLayer.numberOfNodes];
            this.weightChanges = new double[this.numberOfNodes][this.childLayer.numberOfNodes];
            this.biasWeights = new double[this.childLayer.numberOfNodes];
            this.biasValues = new double[this.childLayer.numberOfNodes];
        }

        for(int i = 0; i < this.numberOfNodes; i++){
            this.neuronValues[i] = 0.0;
            this.errors[i] = 0.0;
            if(this.childLayer != null) {
                for (int j = 0; j < this.childLayer.numberOfNodes; j++) {
                    this.weightChanges[i][j] = 0.0;
                }
            }
        }

        if(this.childLayer != null) {
            randomizeWeights();
        }
    }

    private void randomizeWeights(){
        for(int i = 0; i < this.childLayer.numberOfNodes; i++){
            this.biasWeights[i] = Math.random()*2-1;
            this.biasValues[i] = (Math.random()*2-1 < 0)?-1:1;
            for(int j = 0; j < this.numberOfNodes; j++) {
                this.weights[j][i] = Math.random()*2-1;
            }
        }
    }

    public void calculateNeuronValues() {
        double x = 0.0;
        if (parentLayer != null) {
            for (int j=0; j<numberOfNodes; j++) {
                x = 0.0;
                for (int i=0; i<this.parentLayer.numberOfNodes; i++) {
                    x += parentLayer.neuronValues[i] * parentLayer.weights[i][j];
                }
                x += parentLayer.biasValues[j] * parentLayer.biasWeights[j];
                if ((childLayer == null) && linearOutput) {
                    neuronValues[j] = x;
                } else {
                    neuronValues[j] = 1.0 / (1.0 + Math.exp(-x));
                }
            }
        }
    }

    public void adjustWeights() {
        double dw = 0.0;
        if (childLayer != null) {
            for (int i=0; i<numberOfNodes; i++) {
                for (int j=0; j<this.childLayer.numberOfNodes; j++) {
                    dw = learningRate * childLayer.errors[j] * neuronValues[i];
                    if (useMomentum) {
                        weights[i][j] += dw + momentumFactor * weightChanges[i][j];
                        weightChanges[i][j] = dw;
                    } else {
                        weights[i][j] += dw;
                    }
                }
            }
            for (int j=0; j<this.childLayer.numberOfNodes; j++) {
                biasWeights[j] += learningRate * childLayer.errors[j] * biasValues[j];
            }
        }
    }

    public void calculateErrors() {
        double sum = 0.0;
        if (childLayer == null) { //output layer
            for (int i = 0; i < numberOfNodes; i++) {
                errors[i] = (desiredValues[i] - neuronValues[i]) * neuronValues[i] * (1.0 - neuronValues[i]);
            }
        } else if (parentLayer == null) { //input layer
            for (int i = 0; i < numberOfNodes; i++) {
                errors[i] = 0.0f;
            }
        } else { //hidden layer
            for (int i=0; i<numberOfNodes; i++) {
                sum = 0.0;
                for (int j=0; j<this.childLayer.numberOfNodes; j++) {
                    sum += childLayer.errors[j] * weights[i][j];
                }
                errors[i] = sum * neuronValues[i] * (1.0 - neuronValues[i]);
            }
        }
    }
}
