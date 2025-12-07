"""
Streamlit App - Discover + Contribute + Plan
"""

import streamlit as st
import sys
import pickle
import os
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from scraper_maps import MapsLoc
from attribute_classifier import AttributeClassifier
from db import DatabaseManager
from retrieve import VectorStore
from planner import ItineraryPlanner
from config import GROUP_TYPES

@st.cache_resource
def init_components():
    """Initialize and cache all backend components"""
    os.makedirs("models", exist_ok=True)
    components = {}
    
    components.update({
        'db': DatabaseManager(),
        'maps_scraper': MapsLoc(),
        'vector_store': VectorStore(),
        'planner': ItineraryPlanner()
    })

    # 2. Initialize or Load Classifier
    classifier_cache = "models/classifier_cache.pkl"
    classifier = None
    if os.path.exists(classifier_cache):
        try:
            with open(classifier_cache, 'rb') as f:
                classifier = pickle.load(f)
        except Exception as e:
            print(f"⚠️ Cache load failed, creating new: {e}")
            classifier = AttributeClassifier()
    else:
        classifier = AttributeClassifier()

    if not classifier.is_fitted:
        try:
            print("🔄 Training Hybrid Classifier on startup...")
            existing_places = components['db'].get_all_places(limit=100)
            if existing_places:
                classifier.fit_clusters(existing_places)
                with open(classifier_cache, 'wb') as f:
                    pickle.dump(classifier, f)
        except Exception as e:
            print(f"⚠️ Classifier training failed: {e}")
    
    components['classifier'] = classifier
    return components

def main():
    st.set_page_config(page_title="AI Travel Companion", page_icon="🗺️", layout="wide")
    
    components = init_components()

    # Initialize session state variables
    if 'last_search_results' not in st.session_state:
        st.session_state.last_search_results = []
    if 'generated_itinerary' not in st.session_state:
        st.session_state.generated_itinerary = None
    if 'current_query_context' not in st.session_state:
        st.session_state.current_query_context = {}
    if 'query' not in st.session_state:
        st.session_state.query = ""
    if 'location' not in st.session_state:
        st.session_state.location = ""

    with st.sidebar:
        st.title("👤 Travel Profile")
        group_type = st.selectbox("Traveling as...", GROUP_TYPES, index=0)
        
        st.markdown("---")
        
        st.subheader("📜 Recent Searches")
        history = components['db'].get_user_search_history("user_123", limit=5)

        if history:
            for item in history:
                if st.button(f"{item['query']} in {item['location']}", use_container_width=True, key=f"hist_{item['timestamp']}"):
                    st.session_state.query = item['query']
                    st.session_state.location = item['location']
                    st.rerun()
        else:
            st.caption("No recent searches.")

        if st.button("🗑️ Clear History", use_container_width=True):
            components['db'].clear_user_search_history("user_123")
            st.rerun()

    st.title("🗺️ AI Travel Companion")
    tab1, tab2, tab3 = st.tabs(["🔍 Discover", "📅 Plan Trip", "🤝 Contribute"])

    with tab1:
        st.header("Find Your Places")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.session_state.query = st.text_input("What are you looking for?", value=st.session_state.query, placeholder="Hidden waterfalls near Bengaluru")
        with col2:
            st.session_state.location = st.text_input("Location", value=st.session_state.location, placeholder="Bengaluru")

        if st.button("🔍 Search", type="primary"):
            if st.session_state.query and st.session_state.location:
                with st.spinner("Searching..."):
                    user_type = components['classifier'].classify_user_type(st.session_state.query)
                    user_context = {'group_type': group_type, 'user_type': user_type}
                    
                    components['db'].log_user_search({
                        'user_id': "user_123",
                        'query': st.session_state.query,
                        'location': st.session_state.location,
                        'timestamp': datetime.now().isoformat(),
                        'detected_type': user_type
                    })

                    db_results = components['vector_store'].hybrid_search(
                        f"{st.session_state.query} {st.session_state.location}", user_context, n_results=8
                    )
                    
                    final_results = db_results
                    
                    if len(db_results) < 5:
                        if len(db_results) == 0:
                            st.warning("No database match. Searching Google Maps...")
                        else:
                            st.info(f"Found {len(db_results)} in DB, adding more from Google...")

                        raw_places = components['maps_scraper'].search_places(st.session_state.query, st.session_state.location)
                        
                        if raw_places:
                            new_places_added = []
                            for p in raw_places:
                                if any(existing['metadata']['name'] == p['name'] for existing in db_results):
                                    continue
                                
                                p['attributes'] = components['classifier'].classify_with_rules(p)
                                p['place_id'] = f"google_{hash(p['name'])}"
                                
                                components['db'].insert_place(p)
                                components['vector_store'].add_place(p)
                                
                                new_places_added.append({
                                    'id': p['place_id'],
                                    'metadata': {
                                        'name': p['name'], 'rating': str(p['rating']),
                                        'description': p.get('description', ''),
                                        'vibe_tags': ','.join(p['attributes'].get('vibe', [])),
                                        'url': p.get('url'), 'source': p.get('source')
                                    },
                                    'score': 1.0
                                })
                            
                            if new_places_added:
                                final_results.extend(new_places_added)
                                st.success(f"🎉 Added {len(new_places_added)} new places from Google!")

                    if not final_results:
                         st.error("No results found in DB or Google.")
                    
                    st.session_state.last_search_results = final_results
                    st.session_state.current_query_context = user_context

        # Display results
        if st.session_state.last_search_results:
            st.success(f"Found {len(st.session_state.last_search_results)} places")
            for res in st.session_state.last_search_results:
                with st.expander(f"📍 {res['metadata']['name']} ({res['metadata'].get('rating', '-')})"):
                    st.write(res['metadata'].get('description', 'No description'))
                    st.caption(f"Tags: {res['metadata'].get('vibe_tags')}")

    with tab2:
        st.header("⚡ Intelligent Itinerary Builder")
        
        if not st.session_state.last_search_results:
            st.info("👈 Please go to 'Discover' and search for places first!")
        else:
            st.write(f"Planning trip using **{len(st.session_state.last_search_results)} places** found in Discover.")
            
            with st.form("itinerary_form"):
                c1, c2, c3 = st.columns(3)
                with c1:
                    days = st.number_input("Days", min_value=1, max_value=5, value=1)
                with c2:
                    start_time = st.time_input("Start Day", value=datetime.strptime("09:00", "%H:%M").time())
                with c3:
                    end_time = st.time_input("End Day", value=datetime.strptime("19:00", "%H:%M").time())
                
                submit_plan = st.form_submit_button("🚀 Generate Itinerary")

            if submit_plan:
                with st.spinner("🤖 The Strategist is sequencing your trip..."):
                    context = st.session_state.get('current_query_context', {})
                    context['group_type'] = group_type
                    
                    # Format times as strings for the planner
                    s_time_str = start_time.strftime("%H:%M")
                    e_time_str = end_time.strftime("%H:%M")
                    
                    plan = components['planner'].create_itinerary(
                        places=st.session_state.last_search_results,
                        user_context=context,
                        num_days=days,
                        start_time=s_time_str,
                        end_time=e_time_str
                    )
                    st.session_state.generated_itinerary = plan

            # Display Itinerary
            if st.session_state.generated_itinerary:
                plan = st.session_state.generated_itinerary
                st.subheader(f"✨ {plan.get('trip_title', 'Your Trip')}")
                
                for day in plan.get('days', []):
                    with st.container(border=True):
                        st.markdown(f"### 🗓️ Day {day['day']}: {day.get('theme', '')}")
                        
                        for item in day.get('schedule', []):
                            col_time, col_act = st.columns([1, 4])
                            with col_time:
                                st.markdown(f"**{item.get('time')}**")
                            with col_act:
                                st.markdown(f"**{item.get('activity', '')}**")
                                if 'reason' in item:
                                    st.caption(f"{item['reason']}")
                                if 'travel_to_next' in item:
                                    travel = item['travel_to_next']
                                    st.info(f"Travel: {travel['distance']} ({travel['duration']})")

    with tab3:
        st.header("Share Hidden Gems")
        st.markdown("Help us build the best travel database! Add places you know and love.")
        
        with st.form("contribution_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                place_name = st.text_input(" Place Name", placeholder="e.g. Corner House Ice Cream")
                location_area = st.text_input("🗺️ Area/Location", placeholder="e.g. Indiranagar, Bengaluru")
                category = st.selectbox(" Category", [
                    "Nature & Adventure", "Cafe & Food", "Shopping & Malls", 
                    "Culture & Temples", "Entertainment (Movies/Gaming)", "Viewpoint"
                ])
                
            with col2:
                vibes = st.multiselect("Vibes (Select all that apply)", [
                    "Peaceful", "Adventurous", "Romantic", "Crowded", "Family-friendly", 
                    "Instagrammable", "Historic", "Luxury", "Budget-friendly"
                ])
                
                best_time = st.selectbox("Best Time to Visit", [
                    "Morning (Sunrise)", "Afternoon", "Evening (Sunset)", "Night", "All Day"
                ])
                
                rating = st.slider("Your Rating", 1.0, 5.0, 4.5)

            description = st.text_area(
                "Why is it special? (Description)", 
                placeholder="Share your experience... e.g. 'Best chocolate fudge in the city, open till late night.'"
            )
            
            submitted = st.form_submit_button("Submit Contribution")

            if submitted:
                if place_name and description and location_area:
                    manual_attributes = {
                        "vibe": [v.lower() for v in vibes],
                        "best_time": [best_time.split(' ')[0].lower()],
                        "duration": ["2 hours"] 
                    }
                    
                    cat_map = {
                        "Nature & Adventure": "nature",
                        "Cafe & Food": "foodie",
                        "Shopping & Malls": "shopping",
                        "Culture & Temples": "culture",
                        "Entertainment (Movies/Gaming)": "entertainment",
                        "Viewpoint": "viewpoint"
                    }
                    manual_attributes['vibe'].append(cat_map.get(category, 'general'))

                    place_data = {
                        "name": place_name,
                        "address": location_area,
                        "description": description,
                        "rating": rating,
                        "source": "User Contributor",
                        "attributes": manual_attributes,
                        "types": [cat_map.get(category, 'point_of_interest')],
                        "place_id": f"user_{hash(place_name)}"
                    }
                    
                    components['db'].insert_place(place_data)
                    components['vector_store'].add_place(place_data)
                    
                    st.success(f"Thank you! **{place_name}** has been added to our database.")
                    st.balloons()
                else:
                    st.error("⚠️ Please fill in Name, Area, and Description.")


if __name__ == "__main__":
    main()
