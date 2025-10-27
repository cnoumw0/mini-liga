package es.unileon.prg2;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class Equipos {
    private Equipo[] equipos;
    private int numEquipos; // contador actual
    private static final int MAX_EQUIPOS = 2;

    public Equipos(String FILENAME) {
        this.equipos = new Equipo[MAX_EQUIPOS];
        this.numEquipos = 0;
        leerCSV(FILENAME); // rellena el vector equipos
    }

    /** Añadir equipo al vector */
    public void addEquipo(Equipo equipo) {
        if (numEquipos < MAX_EQUIPOS) {
            equipos[numEquipos] = equipo;
            numEquipos++;
        } else {
            System.out.println("⚠️ Ya no caben más equipos en la liga.");
        }
    }

    /** Buscar equipo por nombre (para reutilizar si ya existe) */
    private Equipo buscarEquipo(String nombre) {
        for (int i = 0; i < numEquipos; i++) {
            if (equipos[i].getNombre().equalsIgnoreCase(nombre)) {
                return equipos[i];
            }
        }
        return null;
    }

    /** Leer CSV y rellenar el vector de equipos */
    private void leerCSV(String FILENAME) {
        try (InputStream in = getClass().getResourceAsStream(FILENAME);
             BufferedReader br = new BufferedReader(new InputStreamReader(in))) {

            if (in == null) {
                throw new IllegalStateException("No se encuentra el archivo: " + FILENAME);
            }

            String linea;
            while ((linea = br.readLine()) != null) {
                String[] partes = linea.split(",");
                if (partes.length < 2) continue; // línea mal formada

                String nombreEquipo = partes[0].trim();
                String nombreJugador = partes[1].trim();

                // buscar si el equipo ya existe
                Equipo equipo = buscarEquipo(nombreEquipo);
                if (equipo == null) {
                    equipo = new Equipo(nombreEquipo);
                    addEquipo(equipo);
                }

                // añadir jugador
                equipo.addJugador(new Jugadores(nombreJugador));
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public Equipo[] getEquipos() {
        return equipos;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("=== LaLiga ===\n");
        for (int i = 0; i < numEquipos; i++) {
            sb.append(equipos[i].toString()).append("\n");
        }
        return sb.toString();
    }
}
