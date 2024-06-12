using Microsoft.AspNetCore.Mvc;
using System.Net.Http;
using System.Threading.Tasks;


[ApiController]
[Route("api/[controller]")]
//[EnableCors]
public class WeatherController : ControllerBase
{
    private readonly HttpClient _httpClient;
    //private const string WeatherApiKey = "WEATHER_API_KEY";
    private const string WeatherApiBaseUrl = "http://api.openweathermap.org/geo/1.0/zip";
    private const string CountryCode = "US"; // Set the country code to "US" for United States
    private readonly IConfiguration _configuration;

    public WeatherController(IConfiguration configuration)
    {
        _httpClient = new HttpClient();
        _configuration = configuration;
    }

    [HttpGet]
    public async Task<IActionResult> Get(string zipCode)
    {
        string weatherApiKey = _configuration["WEATHER_API_KEY"];
        string url = $"{WeatherApiBaseUrl}?zip={zipCode},{CountryCode}&appid={weatherApiKey}";

        HttpResponseMessage response = await _httpClient.GetAsync(url);

        if (response.IsSuccessStatusCode)
        {
            string jsonResult = await response.Content.ReadAsStringAsync();
            return Ok(jsonResult);
        }

        return StatusCode((int)response.StatusCode);
    }
}
