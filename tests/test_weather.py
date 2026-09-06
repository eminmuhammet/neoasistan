import asyncio
from unittest.mock import MagicMock, patch

from neo.tools.weather import GetWeatherTool


def _fake_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_get_weather_uses_default_city_when_none_given():
    tool = GetWeatherTool(default_city="Bursa")
    geo_payload = {"results": [{"name": "Bursa", "latitude": 40.19, "longitude": 29.06}]}
    forecast_payload = {
        "current": {
            "temperature_2m": 21.5,
            "relative_humidity_2m": 55,
            "weather_code": 1,
            "wind_speed_10m": 10.0,
        },
        "daily": {
            "time": ["2026-09-06", "2026-09-07"],
            "temperature_2m_max": [25.0, 24.0],
            "temperature_2m_min": [16.0, 15.0],
            "precipitation_probability_max": [10, 40],
            "weather_code": [1, 61],
        },
    }

    with patch("httpx.get", side_effect=[_fake_response(geo_payload), _fake_response(forecast_payload)]):
        result = asyncio.run(tool.run())

    assert result.success
    assert result.data["city"] == "Bursa"
    assert result.data["temperature_c"] == 21.5
    assert result.data["condition"] == "genellikle açık"
    assert result.data["today"]["max_c"] == 25.0
    assert result.data["tomorrow"]["condition"] == "hafif yağmur"


def test_get_weather_unknown_city_returns_error():
    tool = GetWeatherTool(default_city="Bursa")
    with patch("httpx.get", return_value=_fake_response({"results": []})):
        result = asyncio.run(tool.run(city="Bilinmeyenşehir123"))

    assert not result.success
    assert result.error


def test_get_weather_network_error_returns_friendly_message():
    import httpx

    tool = GetWeatherTool(default_city="Bursa")
    with patch("httpx.get", side_effect=httpx.ConnectError("boom")):
        result = asyncio.run(tool.run())

    assert not result.success
    assert result.error
