import java.text.DecimalFormat;

public class ConfusionMatrix {
    private int[][] matrix = new int[10][10];

    public ConfusionMatrix(){
        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                matrix[i][j] = 0;
            }
        }
    }

    public void addEntry(int reference, int prediction){
        matrix[reference][prediction]++;
    }

    public void printToConsole(){
        System.out.print("               p  r  e  d  i  c  t  e  d       v  a  l  u  e \n");
        for(int i = 0; i < 10; i++){
            System.out.print(i+"       ");
        }
        System.out.println();
        for(int i = 0; i < 89; i++){
            System.out.print("-");
        }
        System.out.println();

        char[] lettersTrueValue = new char[]{'t','r','u','e',' ','v','a','l','u','e'};

        int countTotal = 0;
        int countErrors = 0;

        for(int i = 0; i < matrix.length; i++){
            for(int j = 0; j < matrix[i].length; j++){
                String value = ""+matrix[i][j];
                System.out.print(value);
                for(int x = 0; x < 8-value.length(); x++){
                    System.out.print(" ");
                }

                countTotal += matrix[i][j];
                if(i != j){
                    countErrors += matrix[i][j];
                }
            }
            System.out.println("        | <- "+i+" "+lettersTrueValue[i]);
        }

        System.out.println();

        double errorRate = (double)countErrors/(double)countTotal*100;
        double accuracy = 100-errorRate;
        System.out.println("Accuracy = "+new DecimalFormat("#.##").format(accuracy)+"%");
        System.out.println("Error Rate = "+new DecimalFormat("#.##").format(errorRate)+"%");
    }
}
