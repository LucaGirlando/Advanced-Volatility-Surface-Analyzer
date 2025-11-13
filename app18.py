import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import interpolate
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Volatility Surface Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(45deg, #FF6B6B 0%, #4ECDC4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 800;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e86ab;
        border-bottom: 3px solid #FF6B6B;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .arbitrage-alert {
        background-color: #ff4444;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .no-arbitrage {
        background-color: #00C851;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .explanation-note {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 5px;
        padding: 5px;
        background-color: #f8f9fa;
        border-left: 3px solid #4ECDC4;
    }
    .metric-explanation {
        font-size: 0.7rem;
        color: #888;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📊 Advanced Volatility Surface Analyzer</div>', unsafe_allow_html=True)
st.markdown("**Professional-grade implied volatility surface construction and real-time arbitrage detection system**")

# =============================================================================
# ADVANCED VOLATILITY SURFACE ENGINE
# =============================================================================

class AdvancedVolatilitySurface:
    def __init__(self, spot_price, risk_free_rate=0.02, dividend_yield=0.0):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield
        self.option_data = None
        self.surface = None
        self.arbitrage_violations = []
        self.volatility_metrics = {}
        
    def black_scholes_implied_vol(self, S, K, T, r, option_price, option_type='call', q=0.0):
        """Calculate implied volatility using Newton-Raphson method"""
        def black_scholes_price(S, K, T, r, sigma, option_type, q):
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                return S * np.exp(-q*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            else:
                return K * np.exp(-r*T) * norm.cdf(-d2) - S * np.exp(-q*T) * norm.cdf(-d1)
        
        def vega(S, K, T, r, sigma, q):
            d1 = (np.log(S/K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            return S * np.exp(-q*T) * norm.pdf(d1) * np.sqrt(T)
        
        # Newton-Raphson method
        sigma = 0.3  # Initial guess
        tolerance = 1e-6
        max_iterations = 100
        
        for i in range(max_iterations):
            price = black_scholes_price(S, K, T, r, sigma, option_type, q)
            v = vega(S, K, T, r, sigma, q)
            
            if abs(price - option_price) < tolerance:
                return sigma
                
            # Avoid division by zero
            if abs(v) < 1e-10:
                sigma += 0.01
                continue
                
            sigma = sigma - (price - option_price) / v
            
            # Boundary checks
            sigma = max(0.01, min(5.0, sigma))
                
        return sigma
    
    def generate_realistic_market_data(self, strikes, expirations, vol_skew=0.2, term_structure=0.1):
        """Generate realistic market data with proper volatility dynamics"""
        data = []
        for expiry in expirations:
            for strike in strikes:
                # Realistic moneyness calculation
                moneyness = strike / self.spot_price
                days_to_expiry = (expiry - datetime.now()).days
                T = max(days_to_expiry / 365.0, 0.01)
                
                # Sophisticated volatility model
                base_vol = 0.15 + 0.05 * np.exp(-T/0.3)  # Mean-reverting term structure
                skew_effect = vol_skew * (moneyness - 1) * np.exp(-T/0.5)  # Time-decaying skew
                smile_effect = 0.08 * (moneyness - 1)**2  # Quadratic smile
                random_noise = np.random.normal(0, 0.008)  # Market microstructure noise
                
                iv = base_vol + skew_effect + smile_effect + random_noise
                iv = max(0.05, min(0.8, iv))  # Realistic bounds
                
                # Calculate option prices using Black-Scholes
                d1 = (np.log(self.spot_price/strike) + (self.risk_free_rate - self.dividend_yield + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
                d2 = d1 - iv * np.sqrt(T)
                
                call_price = (self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(d1) - 
                            strike * np.exp(-self.risk_free_rate * T) * norm.cdf(d2))
                put_price = (strike * np.exp(-self.risk_free_rate * T) * norm.cdf(-d2) - 
                           self.spot_price * np.exp(-self.dividend_yield * T) * norm.cdf(-d1))
                
                # Add realistic bid-ask spread
                spread = 0.01 * (call_price + put_price)
                call_price += np.random.uniform(-spread/2, spread/2)
                put_price += np.random.uniform(-spread/2, spread/2)
                
                data.append({
                    'strike': strike,
                    'expiry': expiry,
                    'days_to_expiry': days_to_expiry,
                    'moneyness': moneyness,
                    'call_price': max(call_price, 0.01),
                    'put_price': max(put_price, 0.01),
                    'implied_vol': iv,
                    'time_to_expiry': T
                })
        
        self.option_data = pd.DataFrame(data)
        self._calculate_volatility_metrics()
        return self.option_data
    
    def _calculate_volatility_metrics(self):
        """Calculate advanced volatility metrics"""
        if self.option_data is None:
            return
            
        # ATM volatility metrics
        atm_data = self.option_data[np.abs(self.option_data['moneyness'] - 1) < 0.05]
        self.volatility_metrics['atm_vol'] = atm_data['implied_vol'].mean()
        
        # Volatility skew (25-delta put - 25-delta call)
        put_skew_data = self.option_data[self.option_data['moneyness'] < 0.95]
        call_skew_data = self.option_data[self.option_data['moneyness'] > 1.05]
        
        if len(put_skew_data) > 0 and len(call_skew_data) > 0:
            self.volatility_metrics['skew'] = (put_skew_data['implied_vol'].mean() - 
                                             call_skew_data['implied_vol'].mean())
        
        # Term structure slope
        short_term = self.option_data[self.option_data['days_to_expiry'] < 30]
        long_term = self.option_data[self.option_data['days_to_expiry'] > 180]
        
        if len(short_term) > 0 and len(long_term) > 0:
            self.volatility_metrics['term_structure_slope'] = (long_term['implied_vol'].mean() - 
                                                             short_term['implied_vol'].mean())
        
        # Volatility surface curvature
        self.volatility_metrics['surface_curvature'] = self.option_data['implied_vol'].std()
    
    def build_advanced_volatility_surface(self, method='spline', grid_points=50):
        """Build advanced volatility surface with multiple interpolation methods"""
        if self.option_data is None:
            raise ValueError("Option data not available")
            
        strikes = self.option_data['strike'].unique()
        expiries = self.option_data['days_to_expiry'].unique()
        
        # Create interpolation grid
        strike_grid, expiry_grid = np.meshgrid(
            np.linspace(strikes.min(), strikes.max(), grid_points),
            np.linspace(expiries.min(), expiries.max(), grid_points)
        )
        
        points = self.option_data[['strike', 'days_to_expiry']].values
        values = self.option_data['implied_vol'].values
        
        if method == 'spline':
            # Thin-plate spline interpolation
            if len(points) > 3:
                spline = interpolate.Rbf(points[:, 0], points[:, 1], values, 
                                       function='thin_plate', smooth=0.1)
                vol_grid = spline(strike_grid, expiry_grid)
            else:
                vol_grid = np.full_like(strike_grid, values.mean())
        
        elif method == 'linear':
            # Linear interpolation
            interp = interpolate.LinearNDInterpolator(points, values)
            vol_grid = interp(strike_grid, expiry_grid)
            vol_grid = np.nan_to_num(vol_grid, nan=values.mean())
        
        self.surface = {
            'strike_grid': strike_grid,
            'expiry_grid': expiry_grid,
            'vol_grid': vol_grid,
            'moneyness_grid': strike_grid / self.spot_price,
            'interpolation_method': method
        }
        
        return self.surface
    
    def detect_butterfly_arbitrage(self, expiry):
        """Advanced butterfly arbitrage detection with multiple conditions"""
        violations = []
        expiry_data = self.option_data[self.option_data['days_to_expiry'] == expiry]
        
        if len(expiry_data) < 3:
            return violations
            
        strikes = expiry_data['strike'].sort_values().values
        call_prices = expiry_data.set_index('strike')['call_price'].reindex(strikes).values
        
        # Check convexity condition
        for i in range(1, len(strikes) - 1):
            K1, K2, K3 = strikes[i-1], strikes[i], strikes[i+1]
            C1, C2, C3 = call_prices[i-1], call_prices[i], call_prices[i+1]
            
            # Butterfly spread condition
            lambda_val = (K3 - K2) / (K3 - K1)
            butterfly_value = lambda_val * C1 + (1 - lambda_val) * C3
            arbitrage_amount = C2 - butterfly_value
            
            if arbitrage_amount > 0.01:  # Practical threshold
                violations.append({
                    'type': 'Butterfly Arbitrage',
                    'strikes': (K1, K2, K3),
                    'violation_amount': arbitrage_amount,
                    'expiry': expiry,
                    'condition': f'C({K2}) > λC({K1}) + (1-λ)C({K3})',
                    'severity': 'High' if arbitrage_amount > 0.05 else 'Medium'
                })
        
        return violations
    
    def detect_calendar_arbitrage(self, strike):
        """Advanced calendar arbitrage detection"""
        violations = []
        strike_data = self.option_data[self.option_data['strike'] == strike]
        
        if len(strike_data) < 2:
            return violations
            
        strike_data = strike_data.sort_values('days_to_expiry')
        expiries = strike_data['days_to_expiry'].values
        call_prices = strike_data['call_price'].values
        
        # Check monotonicity in time
        for i in range(1, len(expiries)):
            T1, T2 = expiries[i-1], expiries[i]
            C1, C2 = call_prices[i-1], call_prices[i]
            
            # Calendar spread condition (for American options)
            if T2 > T1 and C2 < C1:
                violations.append({
                    'type': 'Calendar Arbitrage',
                    'strike': strike,
                    'expiries': (T1, T2),
                    'violation_amount': C1 - C2,
                    'condition': f'C(T₂) < C(T₁) where T₂ > T₁',
                    'severity': 'High'
                })
        
        return violations
    
    def detect_put_call_parity_violations(self, tolerance=0.05):
        """Detect put-call parity violations with configurable tolerance"""
        violations = []
        
        for expiry in self.option_data['days_to_expiry'].unique():
            expiry_data = self.option_data[self.option_data['days_to_expiry'] == expiry]
            
            for strike in expiry_data['strike'].unique():
                strike_data = expiry_data[expiry_data['strike'] == strike]
                
                if len(strike_data) == 1:
                    continue
                    
                call_price = strike_data['call_price'].iloc[0]
                put_price = strike_data['put_price'].iloc[0]
                T = expiry / 365.0
                
                # Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)
                parity_value = (self.spot_price * np.exp(-self.dividend_yield * T) - 
                              strike * np.exp(-self.risk_free_rate * T))
                actual_difference = call_price - put_price
                discrepancy = abs(actual_difference - parity_value)
                
                if discrepancy > tolerance:
                    violations.append({
                        'type': 'Put-Call Parity Violation',
                        'strike': strike,
                        'expiry': expiry,
                        'discrepancy': discrepancy,
                        'condition': f'|C - P - (S*exp(-qT) - K*exp(-rT))| > {tolerance}',
                        'severity': 'High' if discrepancy > 0.1 else 'Medium'
                    })
        
        return violations
    
    def detect_volatility_arbitrage(self):
        """Detect volatility-based arbitrage opportunities"""
        violations = []
        
        if self.surface is None:
            return violations
            
        # Check for negative variances
        negative_variances = self.surface['vol_grid'] < 0
        if np.any(negative_variances):
            violations.append({
                'type': 'Negative Volatility',
                'description': 'Negative implied volatilities detected in surface',
                'severity': 'Critical',
                'locations': np.sum(negative_variances)
            })
        
        # Check for excessive volatility jumps
        vol_gradients = np.gradient(self.surface['vol_grid'])
        excessive_jumps = np.abs(vol_gradients) > 0.5  # 50% jump threshold
        
        if np.any(excessive_jumps):
            violations.append({
                'type': 'Excessive Volatility Gradient',
                'description': 'Unrealistic volatility jumps in surface',
                'severity': 'Medium',
                'max_gradient': np.max(np.abs(vol_gradients))
            })
        
        return violations
    
    def run_comprehensive_arbitrage_analysis(self):
        """Run complete arbitrage analysis suite"""
        self.arbitrage_violations = []
        
        # Basic arbitrage checks
        for expiry in self.option_data['days_to_expiry'].unique():
            self.arbitrage_violations.extend(self.detect_butterfly_arbitrage(expiry))
        
        for strike in self.option_data['strike'].unique():
            self.arbitrage_violations.extend(self.detect_calendar_arbitrage(strike))
        
        self.arbitrage_violations.extend(self.detect_put_call_parity_violations())
        self.arbitrage_violations.extend(self.detect_volatility_arbitrage())
        
        # Calculate arbitrage metrics
        total_violations = len(self.arbitrage_violations)
        high_severity = len([v for v in self.arbitrage_violations if v.get('severity') == 'High'])
        total_arbitrage_amount = sum([v.get('violation_amount', 0) for v in self.arbitrage_violations])
        
        self.arbitrage_metrics = {
            'total_violations': total_violations,
            'high_severity_violations': high_severity,
            'total_arbitrage_amount': total_arbitrage_amount,
            'arbitrage_intensity': total_violations / len(self.option_data) if len(self.option_data) > 0 else 0
        }
        
        return self.arbitrage_violations

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

with st.sidebar:
    st.markdown("### ⚙️ Market Configuration")
    
    # Market parameters with explanations
    spot_price = st.number_input("Underlying Spot Price", 50.0, 500.0, 100.0, 10.0,
                               help="Current price of the underlying asset")
    st.markdown('<div class="metric-explanation">Current market price of the underlying security</div>', unsafe_allow_html=True)
    
    risk_free_rate = st.number_input("Risk-Free Rate (%)", 0.0, 10.0, 2.0, 0.1) / 100
    st.markdown('<div class="metric-explanation">Annualized risk-free interest rate for pricing</div>', unsafe_allow_html=True)
    
    dividend_yield = st.number_input("Dividend Yield (%)", 0.0, 10.0, 1.0, 0.1) / 100
    st.markdown('<div class="metric-explanation">Continuous dividend yield of the underlying</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Option Parameters")
    
    min_strike = st.number_input("Minimum Strike", 50.0, 200.0, 80.0, 5.0,
                               help="Lowest strike price to include in analysis")
    max_strike = st.number_input("Maximum Strike", 100.0, 500.0, 120.0, 5.0,
                               help="Highest strike price to include in analysis")
    strike_step = st.number_input("Strike Step Size", 1.0, 20.0, 5.0, 1.0)
    
    st.markdown("### ⏰ Expiration Structure")
    
    num_expiries = st.slider("Number of Expirations", 3, 12, 6,
                           help="Total number of expiration dates to analyze")
    min_days = st.number_input("Minimum Days to Expiry", 7, 90, 30, 7)
    max_days = st.number_input("Maximum Days to Expiry", 30, 365, 180, 30)
    
    st.markdown("### 🎯 Volatility Dynamics")
    
    vol_skew = st.slider("Volatility Skew Intensity", 0.0, 0.5, 0.2, 0.05,
                        help="Strength of the volatility skew (put-call asymmetry)")
    term_structure = st.slider("Term Structure Slope", 0.0, 0.3, 0.1, 0.05,
                             help="Slope of the volatility term structure")
    
    st.markdown("### 🔍 Analysis Settings")
    
    interpolation_method = st.selectbox("Interpolation Method", 
                                      ["spline", "linear"],
                                      help="Method for constructing the volatility surface")
    
    arbitrage_tolerance = st.slider("Arbitrage Tolerance", 0.01, 0.1, 0.05, 0.01,
                                  help="Minimum violation amount to flag as arbitrage")
    
    if st.button("🔄 Generate New Surface", use_container_width=True):
        st.session_state.generate_new = True

# =============================================================================
# MAIN ANALYSIS ENGINE
# =============================================================================

# Initialize advanced volatility surface engine
vol_surface = AdvancedVolatilitySurface(spot_price, risk_free_rate, dividend_yield)

# Generate expiration dates
today = datetime.now()
expirations = [today + timedelta(days=int(days)) 
               for days in np.linspace(min_days, max_days, num_expiries)]

# Generate strike prices
strikes = np.arange(min_strike, max_strike + strike_step, strike_step)

# Generate realistic market data
option_data = vol_surface.generate_realistic_market_data(strikes, expirations, vol_skew, term_structure)

# Build volatility surface
surface_data = vol_surface.build_advanced_volatility_surface(interpolation_method)

# Run comprehensive arbitrage analysis
arbitrage_violations = vol_surface.run_comprehensive_arbitrage_analysis()

# =============================================================================
# ADVANCED DASHBOARD
# =============================================================================

# SECTION 1: MARKET OVERVIEW AND KEY METRICS
st.markdown('<div class="section-header">📈 Market Overview & Volatility Metrics</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Spot Price", f"${spot_price:.2f}")
    st.markdown('<div class="metric-explanation">Current underlying price</div>', unsafe_allow_html=True)
    
with col2:
    avg_vol = option_data['implied_vol'].mean()
    st.metric("Average Implied Vol", f"{avg_vol:.2%}")
    st.markdown('<div class="metric-explanation">Mean implied volatility across all options</div>', unsafe_allow_html=True)
    
with col3:
    st.metric("Options Analyzed", len(option_data))
    st.markdown('<div class="metric-explanation">Total number of option contracts</div>', unsafe_allow_html=True)
    
with col4:
    st.metric("Arbitrage Violations", len(arbitrage_violations))
    st.markdown('<div class="metric-explanation">Total arbitrage opportunities detected</div>', unsafe_allow_html=True)

# Volatility metrics expansion
with st.expander("Advanced Volatility Metrics"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("ATM Volatility", f"{vol_surface.volatility_metrics.get('atm_vol', 0):.2%}")
        st.markdown('<div class="metric-explanation">At-the-money implied volatility</div>', unsafe_allow_html=True)
    
    with col2:
        skew = vol_surface.volatility_metrics.get('skew', 0)
        st.metric("Volatility Skew", f"{skew:.4f}")
        st.markdown('<div class="metric-explanation">Put volatility minus call volatility</div>', unsafe_allow_html=True)
    
    with col3:
        term_slope = vol_surface.volatility_metrics.get('term_structure_slope', 0)
        st.metric("Term Structure", f"{term_slope:.4f}")
        st.markdown('<div class="metric-explanation">Long-term vs short-term vol difference</div>', unsafe_allow_html=True)
    
    with col4:
        curvature = vol_surface.volatility_metrics.get('surface_curvature', 0)
        st.metric("Surface Curvature", f"{curvature:.4f}")
        st.markdown('<div class="metric-explanation">Volatility surface standard deviation</div>', unsafe_allow_html=True)

# SECTION 2: 3D VOLATILITY SURFACE VISUALIZATION
st.markdown('<div class="section-header">🌋 3D Volatility Surface Analysis</div>', unsafe_allow_html=True)

fig_3d = go.Figure(data=[go.Surface(
    x=surface_data['moneyness_grid'],
    y=surface_data['expiry_grid'],
    z=surface_data['vol_grid'],
    colorscale='Viridis',
    opacity=0.9,
    contours={
        "x": {"show": True, "start": 0.8, "end": 1.2, "size": 0.1},
        "y": {"show": True, "start": min_days, "end": max_days, "size": 30},
        "z": {"show": True, "start": 0.1, "end": 0.4, "size": 0.05}
    }
)])

fig_3d.update_layout(
    title='Implied Volatility Surface (Moneyness vs Time to Expiry)',
    scene=dict(
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Days to Expiry',
        zaxis_title='Implied Volatility',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    height=600
)

st.plotly_chart(fig_3d, use_container_width=True)
st.markdown('<div class="explanation-note">The 3D surface shows implied volatility as a function of moneyness and time to expiration. A properly shaped surface should be smooth and free of arbitrage opportunities.</div>', unsafe_allow_html=True)

# SECTION 3: VOLATILITY SMILE AND TERM STRUCTURE ANALYSIS
st.markdown('<div class="section-header">😊 Volatility Smile & Term Structure</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    # Volatility smile for selected expiries
    selected_expiries = st.multiselect(
        "Select Expiries for Smile Analysis",
        options=sorted(option_data['days_to_expiry'].unique()),
        default=sorted(option_data['days_to_expiry'].unique())[:3],
        help="Choose expiration dates to compare volatility smiles"
    )
    
    fig_smile = go.Figure()
    for expiry in selected_expiries:
        expiry_data = option_data[option_data['days_to_expiry'] == expiry]
        fig_smile.add_trace(go.Scatter(
            x=expiry_data['moneyness'],
            y=expiry_data['implied_vol'],
            mode='lines+markers',
            name=f'{expiry} days',
            line=dict(width=3)
        ))
    
    fig_smile.update_layout(
        title='Volatility Smile by Expiration',
        xaxis_title='Moneyness (K/S)',
        yaxis_title='Implied Volatility',
        height=400
    )
    st.plotly_chart(fig_smile, use_container_width=True)
    st.markdown('<div class="explanation-note">Volatility smile shows how implied volatility varies with moneyness. A smile indicates higher volatility for OTM options, often seen in equity markets.</div>', unsafe_allow_html=True)

with col2:
    # Term structure analysis
    atm_data = option_data[np.abs(option_data['moneyness'] - 1) < 0.02]
    if not atm_data.empty:
        term_structure = atm_data.groupby('days_to_expiry')['implied_vol'].mean().reset_index()
        
        fig_term = go.Figure()
        fig_term.add_trace(go.Scatter(
            x=term_structure['days_to_expiry'],
            y=term_structure['implied_vol'],
            mode='lines+markers',
            line=dict(color='#FF6B6B', width=4)
        ))
        
        fig_term.update_layout(
            title='ATM Volatility Term Structure',
            xaxis_title='Days to Expiry',
            yaxis_title='Implied Volatility',
            height=400
        )
        st.plotly_chart(fig_term, use_container_width=True)
        st.markdown('<div class="explanation-note">Term structure shows how ATM volatility changes with time to expiration. Upward slope indicates contango, downward indicates backwardation.</div>', unsafe_allow_html=True)

# SECTION 4: COMPREHENSIVE ARBITRAGE DETECTION
st.markdown('<div class="section-header">🔍 Advanced Arbitrage Detection</div>', unsafe_allow_html=True)

if arbitrage_violations:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Arbitrage Violations Summary")
        
        violation_types = [v['type'] for v in arbitrage_violations]
        violation_count = pd.Series(violation_types).value_counts()
        
        fig_violations = go.Figure(data=[go.Bar(
            x=violation_count.index,
            y=violation_count.values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        )])
        
        fig_violations.update_layout(
            title='Arbitrage Violations by Type',
            height=300
        )
        st.plotly_chart(fig_violations, use_container_width=True)
        st.markdown('<div class="explanation-note">Distribution of different arbitrage types detected in the current surface</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Violation Details")
        
        for i, violation in enumerate(arbitrage_violations[:8]):
            severity_color = {
                'Critical': '#FF4444',
                'High': '#FF6B6B', 
                'Medium': '#FFA500',
                'Low': '#FFD700'
            }.get(violation.get('severity', 'Medium'), '#FF6B6B')
            
            st.markdown(f"""
            <div style="background-color: {severity_color}; color: white; padding: 10px; border-radius: 5px; margin: 5px 0;">
            <strong>{violation['type']}</strong> ({violation.get('severity', 'Unknown')})<br>
            {violation.get('condition', '')}<br>
            Amount: ${violation.get('violation_amount', violation.get('discrepancy', 0)):.4f}
            </div>
            """, unsafe_allow_html=True)
        
        if len(arbitrage_violations) > 8:
            st.write(f"... and {len(arbitrage_violations) - 8} more violations")
else:
    st.markdown('<div class="no-arbitrage">✅ No arbitrage violations detected in the current volatility surface</div>', unsafe_allow_html=True)
    st.markdown('<div class="explanation-note">A clean surface indicates proper option pricing and market efficiency. Small violations within tolerance levels are normal due to bid-ask spreads.</div>', unsafe_allow_html=True)

# SECTION 5: ARBITRAGE HEATMAP AND SPATIAL ANALYSIS
st.markdown('<div class="section-header">🔥 Arbitrage Opportunity Heatmap</div>', unsafe_allow_html=True)

# Create arbitrage intensity heatmap
if arbitrage_violations:
    heatmap_data = np.zeros((len(strikes), len(expirations)))
    
    for violation in arbitrage_violations:
        if 'strike' in violation and 'expiry' in violation:
            strike_idx = np.argmin(np.abs(strikes - violation['strike']))
            expiry_idx = np.argmin(np.abs([(e - today).days for e in expirations] - violation['expiry']))
            if strike_idx < len(strikes) and expiry_idx < len(expirations):
                heatmap_data[strike_idx, expiry_idx] += 1
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[e.strftime('%m/%d') for e in expirations],
        y=[f"${s:.0f}" for s in strikes],
        colorscale='Reds',
        showscale=True,
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title='Arbitrage Violation Intensity Heatmap',
        xaxis_title='Expiration Date',
        yaxis_title='Strike Price',
        height=400
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    st.markdown('<div class="explanation-note">Heatmap shows concentration of arbitrage violations across strike and expiration dimensions. Red areas indicate potential trading opportunities.</div>', unsafe_allow_html=True)

# SECTION 6: REAL-TIME MONITORING AND ALERTS
st.markdown('<div class="section-header">⏰ Real-Time Market Monitoring</div>', unsafe_allow_html=True)

if st.checkbox("Enable Real-Time Surface Monitoring", help="Simulate live market data updates"):
    refresh_interval = st.slider("Monitoring Frequency (seconds)", 5, 60, 15)
    
    # Simulate real-time updates
    placeholder = st.empty()
    
    for update_count in range(3):  # Simulate 3 updates
        with placeholder.container():
            # Simulate market movement
            new_spot = spot_price * (1 + np.random.normal(0, 0.008))
            
            # Update surface with new data
            vol_surface.spot_price = new_spot
            new_data = vol_surface.generate_realistic_market_data(strikes, expirations)
            new_surface = vol_surface.build_advanced_volatility_surface()
            new_violations = vol_surface.run_comprehensive_arbitrage_analysis()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Live Spot Price", f"${new_spot:.2f}", 
                         delta=f"{(new_spot - spot_price):.2f}")
            with col2:
                st.metric("Active Violations", len(new_violations))
            with col3:
                st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
            
            # Show new violations
            if new_violations:
                st.warning(f"🚨 New arbitrage opportunities detected: {len(new_violations)}")
                for violation in new_violations[:3]:
                    st.write(f"- {violation['type']} (${violation.get('violation_amount', 0):.3f})")
            
            # Brief pause to simulate real-time
            import time
            time.sleep(refresh_interval)

# SECTION 7: PROFESSIONAL DATA EXPORT AND REPORTING
st.markdown('<div class="section-header">💾 Professional Data Export</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generate Analysis Report"):
        # Create comprehensive report
        report_data = {
            'Analysis Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Underlying Spot': spot_price,
            'Risk-Free Rate': f"{risk_free_rate:.2%}",
            'Options Analyzed': len(option_data),
            'Total Violations': len(arbitrage_violations),
            'ATM Volatility': f"{vol_surface.volatility_metrics.get('atm_vol', 0):.2%}",
            'Volatility Skew': f"{vol_surface.volatility_metrics.get('skew', 0):.4f}",
            'Surface Quality': 'Excellent' if len(arbitrage_violations) == 0 else 'Needs Review'
        }
        
        report_df = pd.DataFrame(list(report_data.items()), columns=['Metric', 'Value'])
        st.dataframe(report_df, use_container_width=True)

with col2:
    # Data export options
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    if st.button(f"💾 Export Volatility Surface Data"):
        # Create export data
        export_df = option_data[['strike', 'expiry', 'days_to_expiry', 'moneyness', 
                               'call_price', 'put_price', 'implied_vol']].copy()
        st.success(f"Data ready for export in {export_format} format")

# Footer with professional disclaimer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8rem;'>
    Advanced Volatility Surface Analyzer | Professional Quantitative Tool<br>
    <em>This tool is for professional use only. Arbitrage opportunities may reflect market frictions rather than true risk-free profits.</em>
</div>
""", unsafe_allow_html=True)

# Hidden detailed explanations expander
with st.expander("📚 Detailed Methodology Explanations"):
    st.markdown("""
    ### Volatility Surface Construction Methodology
    
    **Thin-Plate Spline Interpolation**: 
    - Uses radial basis functions to create smooth surfaces
    - Minimizes bending energy for natural-looking volatility landscapes
    - Handles sparse data points effectively
    
    **Arbitrage Detection Logic**:
    - Butterfly arbitrage: Checks convexity of option prices across strikes
    - Calendar arbitrage: Ensures monotonicity in time dimension  
    - Put-call parity: Validates fundamental option relationships
    - Volatility arbitrage: Detects unrealistic volatility patterns
    
    **Market Realism Features**:
    - Bid-ask spread simulation
    - Volatility skew and smile modeling
    - Term structure dynamics
    - Market microstructure noise
    """)
