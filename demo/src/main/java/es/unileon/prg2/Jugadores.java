package es.unileon.prg2;

public class Jugadores {
    private String nombre;

    public Jugadores(String nombre) {
        this.nombre = nombre;
    }

    public String getNombre() {
        return nombre;
    }
    
    @Override
    public String toString() {
        return nombre;
    }

}
