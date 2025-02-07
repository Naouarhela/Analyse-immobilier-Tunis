from bs4 import BeautifulSoup
import requests
import pandas as pd


def scrape_property_details(page_url):
    # Envoyer la requête GET à l'URL
    response = requests.get(page_url)
    if response.status_code != 200:
        print(f"Erreur lors de la récupération de la page {page_url}. Statut: {response.status_code}")
        return []

    # Analyser le contenu HTML avec le parser HTML par défaut
    soup = BeautifulSoup(response.content, "html.parser")

    # Trouver toutes les annonces
    listings = soup.find_all("li", class_="listingBox w100")
    if not listings:
        print("Aucune annonce trouvée sur la page.")
        return []

    properties = []

    for listing in listings:
        # Extraire les informations nécessaires
        try:
            property_link = listing.find("a", class_="listingLink")["href"] if listing.find("a",
                                                                                            class_="listingLink") else None
            title = listing.find("h2", class_="listingTit").get_text(strip=True) if listing.find("h2",
                                                                                                 class_="listingTit") else "No title"
            location = listing.find("h3", class_="listingH3").get_text(strip=True) if listing.find("h3",
                                                                                                   class_="listingH3") else "No location"
            details = listing.find("h4", class_="listingH4 floatR").get_text(strip=True) if listing.find("h4",
                                                                                                         class_="listingH4 floatR") else "No details"

            # Découper les détails (chambres et superficie)
            details_parts = details.split(",") if details != "No details" else []
            rooms = details_parts[0].strip() if len(details_parts) > 0 else "No rooms"
            total_area = details_parts[1].strip() if len(details_parts) > 1 else "No area"

            price = listing.find("span", class_="priceTag")
            price = price.get_text(strip=True) if price else "Price not listed"

            # Si le prix est "à consulter" ou non indiqué, attribuer 100 000 TND
            if "consulter" in price.lower() or "non indiqué" in price.lower() or "Price not listed" in price:
                price = "100 000 TND"

            # Ajouter les informations dans une liste
            properties.append({
                "Title": title,
                "Link": property_link,
                "Location": location,
                "Rooms": rooms,
                "Total Area": total_area,
                "Price": price,
            })

        except Exception as e:
            print(f"Erreur lors du traitement d'une annonce : {e}")
            continue

    return properties


def save_to_excel(properties, file_name="properties.xlsx"):
    # Sauvegarder les données dans un fichier Excel
    df = pd.DataFrame(properties)
    try:
        df.to_excel(file_name, index=False, engine="openpyxl")
        print(f"Données enregistrées avec succès dans {file_name}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde dans le fichier Excel : {e}")


# URL principale à scraper
page_url = "https://www.mubawab.tn/fr/cc/immobilier-a-vendre-all"
properties = scrape_property_details(page_url)

# Sauvegarder les données dans un fichier Excel
if properties:
    save_to_excel(properties)
else:
    print("Aucune propriété à sauvegarder.")
