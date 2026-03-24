import numpy as np


# ─────────────────────────────────────
# PERCEPTRÓN SIMPLE
# ─────────────────────────────────────
class PerceptronSimple:

    def __init__(self, n_entradas, tasa_aprendizaje=0.1, epocas=20):
        self.pesos = np.zeros(n_entradas + 1)  # +1 bias
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas

    def suma_ponderada(self, entradas):
        return np.dot(entradas, self.pesos[1:]) + self.pesos[0]

    def predecir(self, entradas):
        suma = self.suma_ponderada(entradas)
        return 1 if suma >= 0 else 0

    def entrenar(self, X, y):
        for _ in range(self.epocas):
            for entradas, etiqueta in zip(X, y):

                prediccion = self.predecir(entradas)
                error = etiqueta - prediccion

                # actualizar pesos
                self.pesos[1:] += self.tasa_aprendizaje * error * entradas
                self.pesos[0] += self.tasa_aprendizaje * error


# ─────────────────────────────────────
# PALABRAS CLAVE
# ─────────────────────────────────────
palabras = [
    "factura",     # 0
    "contraseña",  # 1
    "reunion",     # 2
    "oferta",      # 3
    "proyecto",    # 4
    "premio",      # 5
    "descuento"    # 6
]


# ─────────────────────────────────────
# CONVERTIR TEXTO A VECTOR
# ─────────────────────────────────────
def vectorizar(texto):

    texto = texto.lower()

    vector = []
    detectadas = []

    for palabra in palabras:
        if palabra in texto:
            vector.append(1)
            detectadas.append(palabra)
        else:
            vector.append(0)

    return np.array(vector), detectadas


# ─────────────────────────────────────
# DATOS ENTRENAMIENTO P1
# Importante vs Otros
# ─────────────────────────────────────
X_p1 = np.array([

# IMPORTANTES
# reunión + proyecto
[0,0,1,0,1,0,0],

# reunión
[0,0,1,0,0,0,0],

# proyecto
[0,0,0,0,1,0,0],

# factura + reunión
[1,0,1,0,0,0,0],

# factura + proyecto
[1,0,0,0,1,0,0],

# factura + reunión + proyecto
[1,0,1,0,1,0,0],


# NO IMPORTANTES

# oferta + descuento
[0,0,0,1,0,0,1],

# oferta
[0,0,0,1,0,0,0],

# descuento
[0,0,0,0,0,0,1],

# contraseña + premio
[0,1,0,0,0,1,0],

# contraseña
[0,1,0,0,0,0,0],

# premio
[0,0,0,0,0,1,0]

])

y_p1 = np.array([
1,1,1,1,1,1,   # importantes
0,0,0,0,0,0    # otros
])


# ─────────────────────────────────────
# DATOS ENTRENAMIENTO P2
# Promoción vs Spam
# ─────────────────────────────────────
X_p2 = np.array([

# PROMOCIÓN

# oferta + descuento
[0,0,0,1,0,0,1],

# oferta
[0,0,0,1,0,0,0],

# descuento
[0,0,0,0,0,0,1],


# SPAM

# contraseña + premio
[0,1,0,0,0,1,0],

# contraseña
[0,1,0,0,0,0,0],

# premio
[0,0,0,0,0,1,0]

])

y_p2 = np.array([
1,1,1,   # promoción
0,0,0    # spam
])


# ─────────────────────────────────────
# ENTRENAR MODELOS
# ─────────────────────────────────────
p1 = PerceptronSimple(7)
p1.entrenar(X_p1, y_p1)

p2 = PerceptronSimple(7)
p2.entrenar(X_p2, y_p2)


# ─────────────────────────────────────
# MOSTRAR PESOS
# ─────────────────────────────────────
def mostrar_pesos():

    print("\nPESOS PERCEPTRÓN 1 (IMPORTANTE)")
    print("Bias:", p1.pesos[0])

    for i, palabra in enumerate(palabras):
        print(palabra, "=", p1.pesos[i+1])

    print("\nPESOS PERCEPTRÓN 2 (PROMOCIÓN)")
    print("Bias:", p2.pesos[0])

    for i, palabra in enumerate(palabras):
        print(palabra, "=", p2.pesos[i+1])


# ─────────────────────────────────────
# CLASIFICADOR EN CASCADA
# ─────────────────────────────────────
'''def clasificar_correo(texto):

    print("\n===================================")
    print("Correo:", texto)

    vector, detectadas = vectorizar(texto)

    print("Vector:", vector)
    print("Palabras detectadas:", detectadas)

    # P1
    suma1 = p1.suma_ponderada(vector)
    r1 = p1.predecir(vector)

    print("\nPerceptrón 1 (Importante)")
    print("Suma ponderada:", suma1)
    print("Salida:", r1)

    if r1 == 1:
        print("CLASIFICACIÓN FINAL: IMPORTANTE")
        return

    # P2
    suma2 = p2.suma_ponderada(vector)
    r2 = p2.predecir(vector)

    print("\nPerceptrón 2 (Promoción)")
    print("Suma ponderada:", suma2)
    print("Salida:", r2)

    if r2 == 1:
        print("CLASIFICACIÓN FINAL: PROMOCIÓN")
    else:
        print("CLASIFICACIÓN FINAL: SPAM")
 '''
def clasificar_correo(texto):

    vector, detectadas = vectorizar(texto)

    # ─── PERCEPTRÓN 1 ───
    suma1 = p1.suma_ponderada(vector)
    r1 = p1.predecir(vector)

    if r1 == 1:
        return {
            "categoria": "Importante",
            "vector": vector.tolist(),
            "palabras": detectadas,
            "p1": {
                "suma": float(suma1),
                "salida": int(r1)
            },
            "p2": None
        }

    # ─── PERCEPTRÓN 2 ───
    suma2 = p2.suma_ponderada(vector)
    r2 = p2.predecir(vector)

    categoria = "Promoción" if r2 == 1 else "Spam"

    return {
        "categoria": categoria,
        "vector": vector.tolist(),
        "palabras": detectadas,
        "p1": {
            "suma": float(suma1),
            "salida": int(r1)
        },
        "p2": {
            "suma": float(suma2),
            "salida": int(r2)
        }
    }

# ─────────────────────────────────────
# MOSTRAR PESOS
# ─────────────────────────────────────
#mostrar_pesos()


# ─────────────────────────────────────
# PRUEBAS
# ─────────────────────────────────────
#clasificar_correo("Reunión del proyecto mañana")

#clasificar_correo("Oferta especial descuento 50%")

#clasificar_correo("Ganaste premio ahora")

#clasificar_correo("Factura del mes")
