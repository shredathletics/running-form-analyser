# Shred Athletics Running Form Analyser

A professional-grade running form analysis tool that uses AI to provide detailed feedback on your running technique.

## Features

- Real-time running form analysis using MediaPipe pose detection
- Detailed analysis of stride length, knee angles, and posture
- Personalized recommendations for improvement
- Confidence-weighted scoring system
- Cross-platform compatibility

## Live Demo

Access the running form analyser at: [URL will be added after deployment]

## Setup for Local Development

1. Clone the repository:
```bash
git clone [repository-url]
cd running-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start the backend server:
```bash
python -m uvicorn server:app --reload
```

4. Start the frontend server:
```bash
python -m http.server 8080
```

5. Visit `http://127.0.0.1:8080/index.html` in your browser

## Usage

1. Record a video of yourself running (side view recommended)
2. Upload the video through the web interface
3. Receive detailed analysis and recommendations

## Technical Requirements

- Python 3.9+
- Supported video formats: MP4, MOV, AVI
- Maximum file size: 100MB
- Recommended video length: 10-30 seconds
- Clear view of full body while running

## Privacy

- All video processing is done locally
- Videos are not stored after analysis
- No personal data is collected

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For support or queries, please open an issue on GitHub.

## Deployment Instructions

### Backend Deployment (Using Heroku)

1. Create a new Heroku app
2. Set up the Python buildpack
3. Deploy using Git:
```bash
heroku login
git init
git add .
git commit -m "Initial deployment"
heroku git:remote -a your-app-name
git push heroku main
```

### Frontend Deployment (Using GitHub Pages)

1. Create a new repository on GitHub
2. Push your code:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin your-repo-url
git push -u origin main
```

3. Enable GitHub Pages in your repository settings
4. Update the API endpoint in `index.html` to point to your deployed backend

## Custom Domain Setup

1. Purchase a domain (e.g., from Google Domains)
2. Add custom domain in GitHub Pages settings
3. Configure DNS settings:
   - Add an A record pointing to GitHub Pages IP
   - Add a CNAME record for www subdomain
4. Update the backend URL in the frontend code

## Environment Variables

Create a `.env` file with the following variables:
```
PORT=8000
ALLOWED_ORIGINS=https://your-domain.com
```

## Security Considerations

- In production, update CORS settings to only allow your domain
- Implement rate limiting
- Add API key authentication
- Set up SSL certificates

## Support

For support or feature requests, please open an issue on GitHub or contact support@shredathletics.com 