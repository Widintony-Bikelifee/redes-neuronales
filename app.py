from flask import Flask, render_template, request, jsonify
from clasificador import clasificar_correo

app = Flask(__name__)

@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/clasificar", methods=["POST"])
def clasificar():
    data = request.get_json()
    texto = data["texto"]

    resultado = clasificar_correo(texto)

    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True)