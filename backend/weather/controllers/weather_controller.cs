using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Threading.Tasks;

[ApiController]
[Route("api/[controller]")]
[EnableCors]
public class WeatherController : ControllerBase
{
    private readonly HttpClient _httpClient;
    private const string WeatherApiKey = "WEATHER_API_KEY";
    private const string WeatherApiBaseUrl = "http://api.openweathermap.org/geo/1.0/zip";
    private const string CountryCode = "US"; // Set the country code to "US" for United States

    public WeatherController()
    {
        _httpClient = new HttpClient();
    }

    [HttpGet]
    public async Task<IActionResult> Get(string zipCode)
    {
        string url = $"{WeatherApiBaseUrl}?zip={zipCode},{CountryCode}&appid={WeatherApiKey}";

        HttpResponseMessage response = await _httpClient.GetAsync(url);

        if (response.IsSuccessStatusCode)
        {
            string jsonResult = await response.Content.ReadAsStringAsync();
            return Ok(jsonResult);
        }

        return StatusCode((int)response.StatusCode);
    }
}
