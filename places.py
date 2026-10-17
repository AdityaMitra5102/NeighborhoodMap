import requests
import math
import requests

# Cyrillic -> Latin mapping (Macedonian first, plus common Serbian/Russian/Bulgarian)
CYR_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'ѓ': 'gj', 'е': 'e', 'ж': 'zh',
    'з': 'z', 'ѕ': 'dz', 'и': 'i', 'ј': 'j', 'к': 'k', 'л': 'l', 'љ': 'lj', 'м': 'm',
    'н': 'n', 'њ': 'nj', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'ќ': 'kj',
    'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'c', 'ч': 'ch', 'џ': 'dzh', 'ш': 'sh',
    'ђ': 'dj', 'ћ': 'c', 'ё': 'yo', 'й': 'y', 'ы': 'y', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'щ': 'shch', 'ъ': '', 'ь': '', 'і': 'i', 'ї': 'yi', 'є': 'ye', 'ґ': 'g'
}

def to_latin(text: str) -> str:
    """Transliterate Cyrillic characters to Latin equivalent."""
    if not text:
        return ""
    
    chars = list(text)
    result = []
    
    for i, ch in enumerate(chars):
        lower = ch.lower()
        if lower not in CYR_MAP:
            result.append(ch)
            continue
            
        trans = CYR_MAP[lower]
        if not ch.isupper() or not trans:
            result.append(trans)
            continue
            
        prev_is_up = chars[i - 1].isupper() if i > 0 else False
        next_is_up = chars[i + 1].isupper() if i < len(chars) - 1 else False
        
        if prev_is_up or next_is_up:
            result.append(trans.upper())
        else:
            result.append(trans.capitalize())
            
    return "".join(result)

def get_display_name(tags: dict, osm_id_str: str) -> str:
    """Prefer name:en, fallback to transliterated name, or OSM type/id."""
    return tags.get('name:en') or to_latin(tags.get('name', '')) or osm_id_str

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great-circle distance in meters between two lat/lon points."""
    R = 6371000.0
    r_lat1, r_lat2 = math.radians(lat1), math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    
    a = (math.sin(d_lat / 2) ** 2 +
         math.cos(r_lat1) * math.cos(r_lat2) * math.sin(d_lon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))

def get_nearby_places(lat: float, lon: float, radius: int = 500) -> list[dict]:
    """
    Fetch nearby tourism/attraction places from a local Overpass API endpoint
    and apply coordinate validation, untagged node filtering, distance radius checks,
    display name resolution, and distance sorting.
    """
    url = "http://localhost/api/interpreter"
    
    query = f"""[out:json][timeout:60];
(
  nwr["tourism"~"attraction|museum|viewpoint|artwork|gallery|historic"]["name"](around:{radius},{lat},{lon});
);
out center tags;"""

    try:
        response = requests.post(url, data={'data': query})
        response.raise_for_status()
        data = response.json()
        elements = data.get('elements', [])
        
        # Track untagged nodes that only exist as way vertices
        used_by_way = set()
        for e in elements:
            if e.get('type') == 'way' and 'nodes' in e:
                used_by_way.update(e['nodes'])

        places = []
        counts = {}

        for element in elements:
            e_type = element.get('type')
            e_id = element.get('id')
            tags = element.get('tags', {})
            
            # 1. Skip untagged nodes that only exist as way vertices
            if e_type == 'node' and not tags and e_id in used_by_way:
                continue

            # 2. Extract coordinates (node lat/lon vs way/relation center)
            center = element.get('center', {})
            place_lat = element.get('lat', center.get('lat'))
            place_lon = element.get('lon', center.get('lon'))

            # Drop items with missing coordinates
            if place_lat is None or place_lon is None:
                continue

            # Calculate distance from search center
            dist_m = haversine(lat, lon, place_lat, place_lon)

            # 3. Radius Filter: Exclude items exceeding the search radius
            if dist_m > radius:
                continue

            # Build duplicate-handling unique key matching frontend format (e.g. node/123#1)
            osm_ref = f"{e_type}/{e_id}"
            counts[osm_ref] = counts.get(osm_ref, 0) + 1
            key = f"{osm_ref}#{counts[osm_ref]}"

            # Determine display name
            display_name = get_display_name(tags, osm_ref)

            places.append({
                'key': key,
                'id': e_id,
                'type': e_type,
                'name': display_name,
                'tourism': tags.get('tourism'),
                'lat': place_lat,
                'lon': place_lon,
                'distance': round(dist_m, 2),
                'tags': tags
            })

        # 4. Sorting: Sort by distance ascending, then alphabetically by name
        places.sort(key=lambda p: (p['distance'], p['name']))
        return places

    except requests.exceptions.RequestException as e:
        print(f"Error querying Overpass API: {e}")
        return []
        
import wikipediaapi
from wikipediaapi import GeoPoint, Globe, GeoSearchSort

def get_wikipedia_info(
    place_name: str, 
    lat: float = None, 
    lon: float = None, 
    user_agent: str = "PlaceExplorerApp/1.0 (contact@example.com)"
) -> dict | None:
    """
    Finds a Wikipedia page for a given place name without using third-party translation tools.
    Uses Wikipedia's built-in language links and geosearch capabilities.

    :param place_name: Name of the place (in Macedonian or English)
    :param lat: Optional latitude of the place
    :param lon: Optional longitude of the place
    :param user_agent: Required User-Agent for Wikimedia API policy
    :return: Dictionary containing title, summary, url, and language used
    """
    wiki_mk = wikipediaapi.Wikipedia(user_agent=user_agent, language='mk')
    wiki_en = wikipediaapi.Wikipedia(user_agent=user_agent, language='en')


    # Step 3: Geographic Search (if coordinates are provided)
    if lat is not None and lon is not None:
        # Geosearch returns Wikipedia articles near the specified coordinates
        geo_results = wiki_en.geosearch(coord=GeoPoint(lat, lon), sort=GeoSearchSort.DISTANCE, globe=Globe.EARTH, radius=1)
        
        if geo_results:
            # Pick the top nearby result
            first_title = list(geo_results.keys())[0]
            geo_page = wiki_en.page(first_title)
            if geo_page.exists():
                return {
                    "title": geo_page.title,
                    "summary": geo_page.summary,
                    "url": geo_page.fullurl,
                    "lang": "en"
                }

    # Step 1: Check if page exists directly in English
    page_en = wiki_en.page(place_name)
    if page_en.exists():
        return {
            "title": page_en.title,
            "summary": page_en.summary,
            "url": page_en.fullurl,
            "lang": "en"
        }

    # Step 2: Check Macedonian Wikipedia and use Wikipedia's internal interlanguage links
    page_mk = wiki_mk.page(place_name)
    if page_mk.exists():
        # Read Wikipedia's native link to the English version if available
        langlinks = page_mk.langlinks
        if 'en' in langlinks:
            page_en_linked = langlinks['en']
            return {
                "title": page_en_linked.title,
                "summary": page_en_linked.summary,
                "url": page_en_linked.fullurl,
                "lang": "en"
            }
        
        # Fallback to Macedonian content if no English translation link exists
        return {
            "title": page_mk.title,
            "summary": page_mk.summary,
            "url": page_mk.fullurl,
            "lang": "mk"
        }

    
    return None


import re
from urllib.parse import unquote, urlparse
import wikipediaapi


def get_full_wikipedia_text(
    wiki_url: str,
    user_agent: str = "PlaceExplorerApp/1.0 (contact@example.com)",
) -> dict | None:
    """Extracts the complete article text from a Wikipedia URL.

    :param wiki_url: Full Wikipedia article URL (e.g.,
        'https://en.wikipedia.org/wiki/Stone_Bridge_(Skopje)')
    :param user_agent: Required User-Agent header for Wikimedia API policy
    :return: Dictionary with title, full text content, language, and URL
    """
    try:
        parsed_url = urlparse(wiki_url)

        # Extract language code from subdomain (e.g., 'en', 'mk')
        domain_parts = parsed_url.netloc.split(".")
        lang = domain_parts[0] if len(domain_parts) >= 3 else "en"

        # Extract page title from URL path and decode URL-encoded characters
        # Example: /wiki/%D0%9A%D0%B0%D0%BC%D0%B5%D0%BD_%D0%BC%D0%BE%D1%81%D1%82 -> Камен_мост
        path = parsed_url.path
        match = re.search(r"/wiki/(.+)$", path)
        if not match:
            print("Invalid Wikipedia URL format.")
            return None

        raw_title = match.group(1)
        title = unquote(raw_title).replace("_", " ")

        # Initialize API client for the extracted language
        wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)
        page = wiki.page(title)

        if not page.exists():
            print(f"Page '{title}' not found on {lang}.wikipedia.org")
            return None

        return {
            "title": page.title,
            "full_text": page.text,  # Retrieves the entire text content
            "lang": lang,
            "url": page.fullurl,
        }

    except Exception as e:
        print(f"Error processing URL: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    lat_val = 41.99736740449907
    lon_val = 21.407929756047146
    
    results = get_nearby_places(lat_val, lon_val)
    print(len(results))
    for place in results:
        print(f"Name: {place['name']} | Type: {place['tourism']} | Lat: {place['lat']}, Lon: {place['lon']}")
        #info=(get_wikipedia_info(place['name'], lat=place['lat'], lon=place['lon']))
        #print(info)
        #if info:
        #    print(get_full_wikipedia_text(info["url"]))
