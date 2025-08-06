from flask import Flask, request, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import requests
import torch
import webbrowser
import threading
import secrets
import os
import json
from dotenv import load_dotenv

VISTES_PATH = "vistes.json"

def carregar_vistes():
    if os.path.exists(VISTES_PATH):
        with open(VISTES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def desar_vistes(vistes):
    with open(VISTES_PATH, "w", encoding="utf-8") as f:
        json.dump(vistes, f)

# Ruta local del model descarregat prèviament
model_path = "./models"
classifier = None  # Variable global que contindrà el pipeline

# Inicialitza Flask
app = Flask(__name__)
app.secret_key = secrets.token_hex(32)


load_dotenv()

# 🔐 Constants TMDB
API_KEY_TMDB = os.getenv("API_KEY_TMDB")
BASE_URL_TMDB = "https://api.themoviedb.org/3"

GENRES_TMDB = {
    "acció": 28, "aventura": 12, "animació": 16, "comèdia": 35, "crim": 80,
    "documental": 99, "drama": 18, "família": 10751, "fantasia": 14, "història": 36,
    "terror": 27, "música": 10402, "misteri": 9648, "romàntic": 10749,
    "ciència ficció": 878, "cinema de guerra": 10752, "thriller": 53, "western": 37
}
DURACIONS = ["Curta", "Llarga"]
IDIOMES = {
    "ca": "Català",
    "es": "Castellà",
    "en": "Anglès",
    "fr": "Francès",
    "it": "Italià",
    "de": "Alemany"
}

PLATAFORMES = {
    "Netflix": 8,
    "Amazon Prime Video": 9,
    "Disney+": 337,
    "HBO Max": 384,
    "Apple TV+": 350,
    "Movistar Plus": 384  
}

# ✅ Opcional: test de CUDA disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = torch.tensor([1.0, 2.0, 3.0]).to(device)

# 🔁 Funció per carregar el model un sol cop
def carrega_model():
    global classifier
    print("🔁 Carregant model des de:", model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# 🤖 Interpretació de la frase amb IA
def interpreta_frase(frase):
    print("🤖 Interpretant dades amb IA...")
    genere = classifier(frase, list(GENRES_TMDB.keys()))['labels'][0]
    duracio = classifier(frase, DURACIONS)['labels'][0]
    idioma = classifier(frase, IDIOMES)['labels'][0]
    ANYS = ["abans", "despres"]
    any_filtre = classifier(frase, ANYS)['labels'][0]

    return {
        "genere": genere,
        "duracio": duracio,
        "idioma": idioma,
        "any": any_filtre
    }

# 🎬 Crida a l'API de TMDB
def buscar_pellicules(genere_nom, duracio, idioma, any_filtre, plataforma = None):
    genere_id = GENRES_TMDB.get(genere_nom.lower())
    if not genere_id:
        print(f"⚠️ Gènere desconegut: {genere_nom}")
        return []

    url = f"{BASE_URL_TMDB}/discover/movie"
    params = {
        "api_key": API_KEY_TMDB,
        "with_genres": genere_id,
        "language": idioma,
        "sort_by": "popularity.desc",
        "vote_average.gte": 6,
        "vote_count.gte": 50,
        "page": 1,
    }

    

    # Filtre per duració
    if duracio == "curta":
        params["with_runtime.lte"] = 90
    elif duracio == "llarga":
        params["with_runtime.gte"] = 90

    # Filtre per any
    if any_filtre == "abans":
        params["primary_release_date.lte"] = "2009-12-31"
    elif any_filtre == "despres":
        params["primary_release_date.gte"] = "2010-01-01"

    resposta = requests.get(url, params=params)
    if resposta.status_code == 200:
        totes = resposta.json().get("results", [])
        filtrades = [
            peli for peli in totes if peli.get("vote_average", 0) >= 6
        ]
        for item in filtrades:
            credits = obtenir_credits(item["id"])
            item["director"] = credits["director"]
            item["cast"] = credits["cast"]
    if plataforma:
        provider_id = PLATAFORMES.get(plataforma)
        pelis_filtrades = []
        for peli in totes:
            pid = peli["id"]
            r = requests.get(f"{BASE_URL_TMDB}/movie/{pid}/watch/providers", params={"api_key": API_KEY_TMDB})
            info = r.json().get("results", {}).get("ES", {})
            proveidors = info.get("flatrate", [])
            noms_proveidors = [p.get("provider_name") for p in proveidors]
            if plataforma in noms_proveidors:
                pelis_filtrades.append(peli)
        
        return filtrades  
    else:
        print("⚠️ Error a la crida:", resposta.status_code)
        return []
    
def obtenir_credits(movie_id):
    url = f"{BASE_URL_TMDB}/movie/{movie_id}/credits"
    params = {"api_key": API_KEY_TMDB, "language": "ca-ES"}
    resposta = requests.get(url, params=params)
    if resposta.status_code != 200:
        print(f"⚠️ Error cridant credits: {resposta.status_code}")
        return {"director": None, "cast": []}
    
    dades = resposta.json()
    director = None
    for membre in dades.get("crew", []):
        if membre.get("job") == "Director":
            director = membre.get("name")
            break

    # Repartiment: pots limitar a, per exemple, 5 actors principals
    cast = [actor.get("name") for actor in dades.get("cast", [])[:5]]

    return {"director": director, "cast": cast}


# 🌐 Ruta principal
@app.route("/", methods=["GET", "POST"])
def index():
    recomanacions = []
    preferencies = {
        "genres": list(GENRES_TMDB.keys()),
        "idiomes": IDIOMES,
        "plataformes": list(PLATAFORMES.keys())
    }

    # Ruta del fitxer on es guarden les pel·lícules vistes
    fitxer_vistes = "vistes.json"

    # Si no existeix, crea'l buit
    if not os.path.exists(fitxer_vistes):
        with open(fitxer_vistes, "w") as f:
            json.dump([], f)

    # Llegeix els IDs de pel·lícules vistes
    with open(fitxer_vistes, "r") as f:
        vistes = json.load(f)

    if request.method == "POST":
        genere = request.form["genere"]
        duracio = request.form["duracio"]
        idioma = request.form["idioma"]
        any_filtre = request.form["any"]
        plataforma = request.form.get("plataforma", "")

        # Obtenim les recomanacions
        recomanacions = buscar_pellicules(genere, duracio, idioma, any_filtre, plataforma)

        # Obtenir IDs marcats com a vists al formulari
        vistes_formulari = request.form.getlist("vistes")
        vistes_actualitzades = [int(id.strip()) for id in vistes_formulari if id.strip().isdigit()]

# Actualitzar la llista (afegir i eliminar)
        vistes = [v for v in vistes_actualitzades]  # només el que queda marcat

        # Guardem la nova llista al fitxer
        with open(fitxer_vistes, "w", encoding = "utf-8") as f:
            json.dump(vistes, f)
    
    recomanacions.sort(key=lambda item: item.get("id") in vistes)

        # Guardem la nova llista al fitxer
    with open(fitxer_vistes, "w") as f:
        json.dump(vistes, f)
    

    return render_template("index.html", resultats=recomanacions, preferencies=preferencies, vistes=vistes)



# ▶️ Inici de l'aplicació
if __name__ == "__main__":
    carrega_model()
    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5000")).start()
    app.run(debug=False)




