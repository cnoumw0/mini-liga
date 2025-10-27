package es.unileon.prg2;

public class Main {
    public static void main(String[] args) {
        Equipos liga = new Equipos("equipos.csv");
        System.out.println(liga);
    }
}