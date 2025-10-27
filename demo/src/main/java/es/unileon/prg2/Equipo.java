package es.unileon.prg2;

public class Equipo {
    private String nombre;
    private Jugadores[] jugadores;
    private int numJugadores; // contador actual

    private static final int MAX_JUGADORES = 15; 

    public Equipo(String nombre) {
        this.nombre = nombre;
        this.jugadores = new Jugadores[MAX_JUGADORES];
        this.numJugadores = 0;
    }

    public String getNombre() {
        return nombre;
    }

    public Jugadores[] getJugadores() {
        return jugadores;
    }

    public int getNumJugadores() {
        return numJugadores;
    }

    /** Añadir jugador al vector */
    public void addJugador(Jugadores jugador) {
        if (numJugadores < MAX_JUGADORES) {
            jugadores[numJugadores] = jugador;
            numJugadores++;
        } else {
            System.out.println("⚠️ El equipo " + nombre + " ya tiene el máximo de jugadores.");
        }
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Equipo: ").append(nombre).append("\n");
        sb.append("Jugadores:\n");
        for (int i = 0; i < numJugadores; i++) {
            sb.append("  - ").append(jugadores[i].getNombre()).append("\n");
        }
        return sb.toString();
    }
}
