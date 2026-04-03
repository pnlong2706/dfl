// PRT (Prediction-Residual Trust) Module
const PRTManager = (function() {
    const Settings = {
        enabled: false,
        score_type: 'exponential',
        scale: 1.0,
        min_trust: 0.1,
        trust_smoothing: 0.5,
        warmup_rounds: 2,
        apply_to_pseudo: true
    };

    function initialize() {
        const prtSwitch = document.getElementById('prtSwitch');
        const prtSettings = document.getElementById('prtSettings');
        const prtScoreType = document.getElementById('prtScoreType');
        const prtScale = document.getElementById('prtScale');
        const prtMinTrust = document.getElementById('prtMinTrust');
        const prtTrustSmoothing = document.getElementById('prtTrustSmoothing');
        const prtWarmupRounds = document.getElementById('prtWarmupRounds');
        const prtApplyToPseudo = document.getElementById('prtApplyToPseudo');

        if (!prtSwitch || !prtSettings) {
            console.warn('PRT controls not found in DOM');
            return;
        }

        prtSwitch.addEventListener('change', function() {
            Settings.enabled = prtSwitch.checked;
            prtSettings.style.display = prtSwitch.checked ? 'block' : 'none';
            console.log('PRT enabled:', Settings.enabled);
        });

        if (prtScoreType) prtScoreType.addEventListener('change', function() {
            Settings.score_type = prtScoreType.value;
        });
        if (prtScale) prtScale.addEventListener('input', function() {
            Settings.scale = parseFloat(prtScale.value) || 1.0;
        });
        if (prtMinTrust) prtMinTrust.addEventListener('input', function() {
            Settings.min_trust = parseFloat(prtMinTrust.value) || 0.1;
        });
        if (prtTrustSmoothing) prtTrustSmoothing.addEventListener('input', function() {
            Settings.trust_smoothing = parseFloat(prtTrustSmoothing.value) || 0.5;
        });
        if (prtWarmupRounds) prtWarmupRounds.addEventListener('input', function() {
            Settings.warmup_rounds = parseInt(prtWarmupRounds.value) || 2;
        });
        if (prtApplyToPseudo) prtApplyToPseudo.addEventListener('change', function() {
            Settings.apply_to_pseudo = prtApplyToPseudo.checked;
        });

        console.log('PRT controls initialized');
    }

    function getSettings() {
        return {
            enabled: Settings.enabled,
            score_type: Settings.score_type,
            scale: Settings.scale,
            min_trust: Settings.min_trust,
            trust_smoothing: Settings.trust_smoothing,
            warmup_rounds: Settings.warmup_rounds,
            apply_to_pseudo: Settings.apply_to_pseudo
        };
    }

    function isEnabled() {
        return Settings.enabled;
    }

    function setSettings(settings) {
        if (settings) {
            Settings.enabled = settings.enabled || false;
            Settings.score_type = settings.score_type || 'exponential';
            Settings.scale = settings.scale || 1.0;
            Settings.min_trust = settings.min_trust || 0.1;
            Settings.trust_smoothing = settings.trust_smoothing || 0.5;
            Settings.warmup_rounds = settings.warmup_rounds || 2;
            Settings.apply_to_pseudo = settings.apply_to_pseudo !== false;

            const prtSwitch = document.getElementById('prtSwitch');
            const prtSettings = document.getElementById('prtSettings');
            if (prtSwitch) {
                prtSwitch.checked = Settings.enabled;
                if (prtSettings) prtSettings.style.display = Settings.enabled ? 'block' : 'none';
            }
            const prtScoreType = document.getElementById('prtScoreType');
            const prtScale = document.getElementById('prtScale');
            const prtMinTrust = document.getElementById('prtMinTrust');
            const prtTrustSmoothing = document.getElementById('prtTrustSmoothing');
            const prtWarmupRounds = document.getElementById('prtWarmupRounds');
            const prtApplyToPseudo = document.getElementById('prtApplyToPseudo');
            if (prtScoreType) prtScoreType.value = Settings.score_type;
            if (prtScale) prtScale.value = Settings.scale;
            if (prtMinTrust) prtMinTrust.value = Settings.min_trust;
            if (prtTrustSmoothing) prtTrustSmoothing.value = Settings.trust_smoothing;
            if (prtWarmupRounds) prtWarmupRounds.value = Settings.warmup_rounds;
            if (prtApplyToPseudo) prtApplyToPseudo.checked = Settings.apply_to_pseudo;
        }
    }

    function reset() {
        Settings.enabled = false;
        Settings.score_type = 'exponential';
        Settings.scale = 1.0;
        Settings.min_trust = 0.1;
        Settings.trust_smoothing = 0.5;
        Settings.warmup_rounds = 2;
        Settings.apply_to_pseudo = true;

        const prtSwitch = document.getElementById('prtSwitch');
        const prtSettings = document.getElementById('prtSettings');
        if (prtSwitch) prtSwitch.checked = false;
        if (prtSettings) prtSettings.style.display = 'none';
        const prtScoreType = document.getElementById('prtScoreType');
        const prtScale = document.getElementById('prtScale');
        const prtMinTrust = document.getElementById('prtMinTrust');
        const prtTrustSmoothing = document.getElementById('prtTrustSmoothing');
        const prtWarmupRounds = document.getElementById('prtWarmupRounds');
        const prtApplyToPseudo = document.getElementById('prtApplyToPseudo');
        if (prtScoreType) prtScoreType.value = 'exponential';
        if (prtScale) prtScale.value = 1.0;
        if (prtMinTrust) prtMinTrust.value = 0.1;
        if (prtTrustSmoothing) prtTrustSmoothing.value = 0.5;
        if (prtWarmupRounds) prtWarmupRounds.value = 2;
        if (prtApplyToPseudo) prtApplyToPseudo.checked = true;
    }

    return {
        initialize,
        getSettings,
        isEnabled,
        setSettings,
        reset
    };
})();

export default PRTManager;
