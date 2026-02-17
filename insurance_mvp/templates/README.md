# Insurance Claim Review Web UI

Production-quality web interface for AI-powered insurance claim review system.

## Overview

This web UI provides a professional, responsive interface for insurance claim reviewers to:
- View and prioritize pending claims in a review queue
- Review AI assessments with video evidence
- Make decisions (APPROVE/REJECT/REQUEST_MORE_INFO)
- Override AI predictions when necessary
- Monitor system metrics and performance

## Features

### 1. Review Queue (`/queue`)
- **Priority-based sorting**: URGENT → STANDARD → LOW
- **Real-time filtering**: Filter by priority, search by claim ID
- **Auto-refresh**: Queue updates every 30 seconds
- **Status indicators**: Visual severity badges, confidence meters, fraud risk scores
- **Responsive design**: Works on desktop, tablet, and mobile

### 2. Claim Review Page (`/review/{claim_id}`)
- **Video player**: Video.js-powered player with danger clip markers
- **AI Assessment Panel**:
  - Severity with confidence meter
  - Conformal prediction set (90% confidence interval)
  - Review priority (URGENT/STANDARD/LOW)
  - Causal reasoning from AI
- **Fault Assessment**:
  - Fault ratio (0-100%)
  - Scenario type (rear_end, head_on, etc.)
  - Applicable traffic rules
  - Traffic signal and right-of-way context
- **Fraud Risk Analysis**:
  - Risk score (0-100%)
  - Fraud indicators list
  - Visual risk gauge
- **Hazards Timeline**:
  - Detected hazards with timestamps
  - Actor types (car, pedestrian, bicycle)
  - Spatial relationships
- **Evidence Clips**:
  - Keyframe thumbnails
  - Timestamps with descriptions
  - Click to jump to video position
- **Human Decision Form**:
  - Decision buttons (APPROVE/REJECT/REQUEST_MORE_INFO)
  - Override fields (severity, fault ratio, fraud flag)
  - Reasoning text area (required for audit trail)
  - Comments field (optional)

### 3. Metrics Dashboard (`/metrics`)
- **Key Performance Indicators**:
  - Processing rate (claims/hour)
  - Queue depth by priority
  - Average review time
  - AI accuracy (agreement with human reviewers)
- **Charts**:
  - Severity distribution (doughnut chart)
  - Processing volume trend (line chart)
  - AI accuracy trend (line chart)
  - Decision distribution (bar chart)
- **System Health**:
  - API health status
  - Model inference time
  - Queue capacity
  - GPU utilization
- **Recent Activity**:
  - Last 10 reviews with timestamps
  - Reviewer IDs and decisions
  - Review duration

## Design Principles

### Professional Insurance Industry Aesthetic
- **Clean layout**: Minimalist design with clear information hierarchy
- **Inter font**: Professional sans-serif typeface
- **Consistent color scheme**:
  - Blue (#3b82f6): Primary actions, navigation
  - Green (#10b981): Approvals, low risk
  - Yellow/Orange (#f59e0b): Warnings, medium risk
  - Red (#ef4444): Rejections, high risk, urgent priority
  - Gray scale: Neutral elements, backgrounds

### Accessibility (WCAG 2.1 Compliant)
- **Keyboard navigation**: Full keyboard support with shortcuts
- **Screen reader support**: Semantic HTML with ARIA labels
- **Skip to main content**: Quick navigation link
- **High contrast**: Sufficient color contrast ratios
- **Focus indicators**: Visible focus states for all interactive elements

### Performance
- **Fast loading**: Minimal dependencies, CDN-hosted assets
- **Efficient rendering**: Template caching, optimized DOM
- **Progressive enhancement**: Works without JavaScript (basic functionality)
- **Lazy loading**: Images and charts load on demand

### Mobile Responsive
- **Fluid grid layout**: Adapts to screen size
- **Touch-friendly**: Large tap targets (min 44x44px)
- **Optimized typography**: Readable font sizes
- **Collapsed navigation**: Mobile menu for small screens

## Technology Stack

- **Frontend**:
  - TailwindCSS 3.x (CDN) - Utility-first CSS framework
  - Video.js 8.x - HTML5 video player
  - Chart.js 4.x - Data visualization
  - Vanilla JavaScript - No heavy frameworks

- **Backend**:
  - FastAPI - Python async web framework
  - Jinja2 - Server-side templating
  - SQLAlchemy - Database ORM

- **Fonts & Icons**:
  - Inter (Google Fonts) - Primary typeface
  - Heroicons (inline SVG) - Icon system

## Usage

### Setup

1. **Install dependencies** (if not already installed):
   ```bash
   pip install -e ".[all]"
   ```

2. **Start the API server**:
   ```bash
   python -m insurance_mvp.api.main
   ```

3. **Open browser**:
   ```
   http://localhost:8000/queue
   ```

### Integration with Existing API

The web UI routes are defined in `api/web_routes.py` and should be mounted in the main FastAPI app:

```python
from insurance_mvp.api.web_routes import router as web_router

app.include_router(web_router)
```

### Environment Variables

- `DATABASE_URL`: Database connection string (default: sqlite:///./insurance.db)
- `UPLOAD_DIR`: Video upload directory (default: ./data/uploads)
- `DEV_MODE`: Enable development mode (default: true)

## Keyboard Shortcuts

### Queue Page
- `Cmd/Ctrl + K`: Focus search bar
- `Escape`: Clear search

### Review Page
- `Cmd/Ctrl + 1`: Select APPROVE
- `Cmd/Ctrl + 2`: Select REJECT
- `Cmd/Ctrl + 3`: Select REQUEST_MORE_INFO
- `Space`: Play/pause video

## API Endpoints Used

The web UI consumes these API endpoints:

- `GET /claims/{claim_id}/assessment` - Get AI assessment
- `POST /reviews` - Submit review decision
- `GET /api/metrics` - Get system metrics
- `GET /reviews/queue` - Get review queue

## Customization

### Branding
Edit `templates/base.html` to customize:
- Logo and brand name
- Color scheme (TailwindCSS classes)
- Footer links

### Thresholds
Edit confidence/risk thresholds in template files:
- Fraud risk: HIGH (>0.7), MEDIUM (>0.4), LOW (<=0.4)
- Confidence: Colors based on percentage
- Priority: URGENT (red), STANDARD (yellow), LOW (green)

## Production Deployment

### Security Checklist
- [ ] Enable HTTPS (TLS/SSL)
- [ ] Add authentication (OAuth2, SAML)
- [ ] Implement CSRF protection
- [ ] Add rate limiting
- [ ] Enable audit logging
- [ ] Configure CORS properly
- [ ] Use environment-specific configs
- [ ] Sanitize user inputs
- [ ] Add session management
- [ ] Implement role-based access control (RBAC)

### Performance Optimization
- [ ] Enable template caching
- [ ] Use CDN for static assets
- [ ] Enable gzip compression
- [ ] Implement response caching
- [ ] Optimize database queries
- [ ] Add database connection pooling
- [ ] Use Redis for session storage
- [ ] Enable HTTP/2

### Monitoring
- [ ] Add error tracking (Sentry)
- [ ] Implement analytics (PostHog, Mixpanel)
- [ ] Set up uptime monitoring
- [ ] Configure log aggregation
- [ ] Add performance monitoring (New Relic, DataDog)

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+
- Mobile Safari (iOS 14+)
- Chrome Mobile

## Troubleshooting

### Video not playing
- Check video file format (MP4 recommended)
- Verify file path in database
- Ensure upload directory is accessible
- Check browser console for errors

### Templates not rendering
- Verify template directory path in `web_routes.py`
- Check Jinja2 template syntax
- Review FastAPI logs for errors

### Slow performance
- Check database query performance
- Enable template caching
- Optimize chart rendering
- Review network waterfall in DevTools

## License

Proprietary - Internal use only

---

**Built with** ❤️ **for insurance claim reviewers**
