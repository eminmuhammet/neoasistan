from __future__ import annotations

import asyncio
import logging

from .base import RiskLevel, Tool, ToolResult

logger = logging.getLogger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_WEATHER_CODES = {
    0: "açık",
    1: "genellikle açık",
    2: "parçalı bulutlu",
    3: "çok bulutlu",
    45: "sisli",
    48: "kırağı sisi",
    51: "hafif çiseleme",
    53: "çiseleme",
    55: "yoğun çiseleme",
    56: "hafif donan çiseleme",
    57: "yoğun donan çiseleme",
    61: "hafif yağmur",
    63: "yağmur",
    65: "kuvvetli yağmur",
    66: "hafif donan yağmur",
    67: "kuvvetli donan yağmur",
    71: "hafif kar yağışı",
    73: "kar yağışı",
    75: "kuvvetli kar yağışı",
    77: "kar taneleri",
    80: "hafif sağanak yağmur",
    81: "sağanak yağmur",
    82: "şiddetli sağanak yağmur",
    85: "hafif sağanak kar",
    86: "kuvvetli sağanak kar",
    95: "gök gürültülü fırtına",
    96: "dolu ile gök gürültülü fırtına",
    99: "kuvvetli dolu ile gök gürültülü fırtına",
}


def _describe_code(code: int | None) -> str:
    if code is None:
        return "bilinmeyen hava durumu"
    return _WEATHER_CODES.get(code, "bilinmeyen hava durumu")


class GetWeatherTool(Tool):
    name = "get_weather"
    description = (
        "Bir şehir için güncel hava durumunu ve yarınki tahmini döndürür. "
        "'city' verilmezse kullanıcının varsayılan şehri kullanılır."
    )
    risk = RiskLevel.LOW
    input_schema = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "Hava durumu istenen şehir (ör. 'Bursa'). Boş bırakılırsa "
                    "kullanıcının varsayılan şehri kullanılır."
                ),
            }
        },
    }

    def __init__(self, default_city: str) -> None:
        self._default_city = default_city

    async def run(self, city: str | None = None, **kwargs: object) -> ToolResult:
        target_city = (city or self._default_city).strip()
        if not target_city:
            return ToolResult(success=False, error="Hangi şehir için baktığımı bilmiyorum.")
        return await asyncio.to_thread(self._fetch, target_city)

    def _fetch(self, city: str) -> ToolResult:
        import httpx

        try:
            geo_response = httpx.get(
                _GEOCODING_URL, params={"name": city, "count": 1, "language": "tr"}, timeout=8.0
            )
            geo_response.raise_for_status()
            results = geo_response.json().get("results")
            if not results:
                return ToolResult(success=False, error=f"'{city}' adında bir yer bulamadım.")
            location = results[0]

            forecast_response = httpx.get(
                _FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
                    "daily": (
                        "temperature_2m_max,temperature_2m_min,"
                        "precipitation_probability_max,weather_code"
                    ),
                    "timezone": "auto",
                    "forecast_days": 2,
                },
                timeout=8.0,
            )
            forecast_response.raise_for_status()
            payload = forecast_response.json()
        except httpx.HTTPError:
            logger.exception("Hava durumu alınamadı")
            return ToolResult(success=False, error="Hava durumu bilgisine şu an ulaşamıyorum.")
        except Exception:
            logger.exception("Hava durumu işlenirken hata")
            return ToolResult(success=False, error="Hava durumu bilgisi işlenirken bir hata oluştu.")

        current = payload.get("current", {})
        daily = payload.get("daily", {})

        data: dict[str, object] = {
            "city": location.get("name", city),
            "temperature_c": current.get("temperature_2m"),
            "condition": _describe_code(current.get("weather_code")),
            "humidity_percent": current.get("relative_humidity_2m"),
            "wind_kmh": current.get("wind_speed_10m"),
        }

        days = daily.get("time") or []
        if len(days) >= 1:
            data["today"] = {
                "max_c": daily["temperature_2m_max"][0],
                "min_c": daily["temperature_2m_min"][0],
                "rain_chance_percent": daily["precipitation_probability_max"][0],
            }
        if len(days) >= 2:
            data["tomorrow"] = {
                "condition": _describe_code(daily["weather_code"][1]),
                "max_c": daily["temperature_2m_max"][1],
                "min_c": daily["temperature_2m_min"][1],
                "rain_chance_percent": daily["precipitation_probability_max"][1],
            }

        return ToolResult(success=True, data=data)
