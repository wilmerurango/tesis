
public class Wilmer{
    public static void main(String args[]){
        
        // Scanner entrada = new Scanner(System.in);
        // String nombre = "";
        // int num1 = 0, num2 = 0, resultado = 0, parametro= 2;

        // System.out.println("Qual es tu nombre ?: ");
        // nombre = entrada.nextLine();

        // System.out.println("Dame el primer numero: ");
        // num1 = entrada.nextInt();

        // System.out.println("Dame el segundo numero: ");
        // num2 = entrada.nextInt();

        // resultado = num1 + num2;

        // System.out.println("Nombre: " + nombre);

        // if(resultado <= 5){
        //     System.out.println("Perdiste con nota: " + resultado);
        // }else if(resultado <= 7){
        //     System.out.println("Apenas y ganaste Perdiste con nota:  " + resultado);
        // }else{
        //     System.out.println("Ganaste sobrao con valor de: " + resultado);
        // }

        // switch(parametro){
        //     case 1: System.out.println("Perdiste con nota: " + resultado);
        //             break;
        //     case 2: System.out.println("Perdiste con nota: " + resultado);
        //             break;
        //     default: System.out.println("Perdiste con nota: " + resultado);
        //             break;

        // }

        for (int i=1;i<=5;i++){
            System.out.print(i+",");
        }

        int j = 1;
        while (j<=5) { 
            System.out.println(j);
            j += 1;
        }

        int k = -1000;
        do{
            System.out.println(k);
            k -= 200;
        }while(k>=0);

    }
}