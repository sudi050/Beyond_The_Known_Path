"""
LLM-based itinerary planner - The Strategist (Cloud + Local Fallback)
"""
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict
from openai import OpenAI
from config import OPENAI_API_KEY
from scraper_maps import MapsLoc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ItineraryPlanner:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
            except:
                self.client = None
        self.maps_scraper = MapsLoc()

    def create_itinerary(self, places: List[Dict], user_context: Dict, num_days: int = 1, start_time: str = "09:00", end_time: str = "18:00") -> Dict:
        clean_places = [p for p in places if (p.get('name') or p.get('metadata', {}).get('name'))]
        if not clean_places: return {"error": "No valid places found."}

        places_summary = self._prepare_places_summary(clean_places)
        prompt = self._build_planning_prompt(places_summary, user_context, num_days, start_time, end_time)

        raw_plan = {}
        try:
            if not self.client:
                raise ValueError("No OpenAI Key")
            
            logger.info("🧠 Calling OpenAI GPT-4o...")
            raw_plan = self._call_cloud_llm(prompt)
            
        except Exception as e:
            logger.warning(f"⚠️ OpenAI failed ({e}). Falling back to Local TinyLlama logic.")
            raw_plan = self._call_tinyllama_fallback(places_summary, num_days, start_time)

        return self._execute_plan_logistics(raw_plan, clean_places)

    def _call_cloud_llm(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only travel planner."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _call_tinyllama_fallback(self, places: List[Dict], num_days: int, start_time_str: str) -> Dict:
        """
        Simulates what TinyLlama would output for an itinerary.
        Constructs a valid JSON plan using algorithmic sorting.
        """
        logger.info("🦙 Running Local Planner (TinyLlama Simulation)...")
        
        sorted_places = sorted(places, key=lambda x: x.get('rating', 0), reverse=True)
        
        itinerary = {
            "trip_title": "My Local Adventure (Offline Mode)",
            "days": []
        }
        
        try:
            current_time = datetime.strptime(start_time_str, "%H:%M")
        except:
            current_time = datetime.strptime("09:00", "%H:%M")

        places_per_day = max(1, len(sorted_places) // num_days)
        
        for day_num in range(1, num_days + 1):
            day_schedule = []
            start_idx = (day_num - 1) * places_per_day
            day_places = sorted_places[start_idx : start_idx + places_per_day]
            
            for p in day_places:
                day_schedule.append({
                    "time": current_time.strftime("%H:%M"),
                    "place_id": p['id'],
                    "activity": f"Visit {p['name']}",
                    "reason": "Top rated local spot"
                })
                current_time += timedelta(hours=2)
                
                if current_time.hour >= 13 and current_time.hour < 14:
                    day_schedule.append({
                        "time": "13:00",
                        "activity": "Lunch Break",
                        "reason": "Local cuisine"
                    })
                    current_time += timedelta(hours=1)

            itinerary["days"].append({
                "day": day_num,
                "theme": "Local Highlights",
                "schedule": day_schedule
            })
            
            current_time = datetime.strptime(start_time_str, "%H:%M")

        return itinerary

    def _prepare_places_summary(self, places: List[Dict]) -> List[Dict]:
        summaries = []
        for i, place in enumerate(places):
            meta = place.get('metadata', {})
            name = meta.get('name') or place.get('name') or f"Place {i}"
            
            summaries.append({
                'id': i,
                'name': name,
                'rating': meta.get('rating') or place.get('rating') or 0
            })
        return summaries

    def _build_planning_prompt(self, places, user_context, num_days, start, end) -> str:
        return f"""
        You are an expert travel logistician. Create a detailed {num_days}-day itinerary.
        
        USER CONTEXT:
        - Traveler Type: {user_context.get('user_type', 'General')}
        - Group: {user_context.get('group_type', 'Solo')}
        - Daily Schedule: {start} to {end}
        
        AVAILABLE PLACES (Select the best ones):
        {json.dumps(places, indent=2)}
        
        RULES:
        1. Group geographically close places.
        2. Include meal breaks.
        3. Output MUST be valid JSON.
        
        JSON FORMAT:
        {{
            "trip_title": "Creative name for the trip",
            "days": [
                {{
                    "day": 1,
                    "theme": "Theme Name",
                    "schedule": [
                        {{
                            "time": "09:00",
                            "place_id": 0,
                            "activity": "Visit",
                            "reason": "Why this fits here"
                        }}
                    ]
                }}
            ]
        }}
        """

    def _call_llm_planner(self, prompt: str) -> Dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a JSON-only API."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def _execute_plan_logistics(self, plan: Dict, original_places: List[Dict]) -> Dict:
        if 'days' not in plan: return plan
        
        for day in plan['days']:
            schedule = day.get('schedule', [])
            for i in range(len(schedule) - 1):
                curr, next_stop = schedule[i], schedule[i+1]
                if 'place_id' in curr and 'place_id' in next_stop:
                    try:
                        idx1, idx2 = int(curr['place_id']), int(next_stop['place_id'])
                        if idx1 < len(original_places) and idx2 < len(original_places):
                            p1 = original_places[idx1]
                            p2 = original_places[idx2]
                            n1 = p1.get('name') or p1.get('metadata', {}).get('name')
                            n2 = p2.get('name') or p2.get('metadata', {}).get('name')
                            if n1 and n2:
                                dirs = self.maps_scraper.get_directions(n1, n2)
                                if dirs:
                                    curr['travel_to_next'] = dirs
                    except: pass
        return plan
