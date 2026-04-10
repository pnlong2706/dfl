// PRT (Prediction-Residual Trust) Module
const PRTManager = (function() {
    const Settings = {
        enabled: false,
        score_type: 'exponential',
        scale: 1.0,
        min_trust: 0.1,
        trust_smoothing: 0.5,
        warmup_rounds: 2,
        apply_to_pseudo: true,
        adaptive: true,
        exclusion_z: 2.5,
        direction_check: true,
        direction_penalty: 0.3
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
        const prtAdaptive = document.getElementById('prtAdaptive');
        const prtAdaptiveSettings = document.getElementById('prtAdaptiveSettings');
        const prtExclusionZ = document.getElementById('prtExclusionZ');
        const prtDirectionCheck = document.getElementById('prtDirectionCheck');
        const prtDirectionPenalty = document.getElementById('prtDirectionPenalty');

        if (!prtSwitch || !prtSettings) {
            console.warn('PRT controls not found in DOM');
            return;
        }

        prtSwitch.addEventListener('change', function() {
            Settings.enabled = prtSwitch.checked;
            prtSettings.style.display = prtSwitch.checked ? 'block' : 'none';
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
        if (prtAdaptive) prtAdaptive.addEventListener('change', function() {
            Settings.adaptive = prtAdaptive.checked;
            if (prtAdaptiveSettings) prtAdaptiveSettings.style.display = prtAdaptive.checked ? 'block' : 'none';
        });
        if (prtExclusionZ) prtExclusionZ.addEventListener('input', function() {
            Settings.exclusion_z = parseFloat(prtExclusionZ.value) || 2.5;
        });
        if (prtDirectionCheck) prtDirectionCheck.addEventListener('change', function() {
            Settings.direction_check = prtDirectionCheck.checked;
        });
        if (prtDirectionPenalty) prtDirectionPenalty.addEventListener('input', function() {
            Settings.direction_penalty = parseFloat(prtDirectionPenalty.value) || 0.3;
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
            apply_to_pseudo: Settings.apply_to_pseudo,
            adaptive: Settings.adaptive,
            exclusion_z: Settings.exclusion_z,
            direction_check: Settings.direction_check,
            direction_penalty: Settings.direction_penalty
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
            Settings.adaptive = settings.adaptive !== false;
            Settings.exclusion_z = settings.exclusion_z || 2.5;
            Settings.direction_check = settings.direction_check !== false;
            Settings.direction_penalty = settings.direction_penalty || 0.3;

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
            const prtAdaptive = document.getElementById('prtAdaptive');
            const prtAdaptiveSettings = document.getElementById('prtAdaptiveSettings');
            const prtExclusionZ = document.getElementById('prtExclusionZ');
            const prtDirectionCheck = document.getElementById('prtDirectionCheck');
            const prtDirectionPenalty = document.getElementById('prtDirectionPenalty');
            if (prtScoreType) prtScoreType.value = Settings.score_type;
            if (prtScale) prtScale.value = Settings.scale;
            if (prtMinTrust) prtMinTrust.value = Settings.min_trust;
            if (prtTrustSmoothing) prtTrustSmoothing.value = Settings.trust_smoothing;
            if (prtWarmupRounds) prtWarmupRounds.value = Settings.warmup_rounds;
            if (prtApplyToPseudo) prtApplyToPseudo.checked = Settings.apply_to_pseudo;
            if (prtAdaptive) prtAdaptive.checked = Settings.adaptive;
            if (prtAdaptiveSettings) prtAdaptiveSettings.style.display = Settings.adaptive ? 'block' : 'none';
            if (prtExclusionZ) prtExclusionZ.value = Settings.exclusion_z;
            if (prtDirectionCheck) prtDirectionCheck.checked = Settings.direction_check;
            if (prtDirectionPenalty) prtDirectionPenalty.value = Settings.direction_penalty;
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
        Settings.adaptive = true;
        Settings.exclusion_z = 2.5;
        Settings.direction_check = true;
        Settings.direction_penalty = 0.3;

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
        const prtAdaptive = document.getElementById('prtAdaptive');
        const prtAdaptiveSettings = document.getElementById('prtAdaptiveSettings');
        const prtExclusionZ = document.getElementById('prtExclusionZ');
        const prtDirectionCheck = document.getElementById('prtDirectionCheck');
        const prtDirectionPenalty = document.getElementById('prtDirectionPenalty');
        if (prtScoreType) prtScoreType.value = 'exponential';
        if (prtScale) prtScale.value = 1.0;
        if (prtMinTrust) prtMinTrust.value = 0.1;
        if (prtTrustSmoothing) prtTrustSmoothing.value = 0.5;
        if (prtWarmupRounds) prtWarmupRounds.value = 2;
        if (prtApplyToPseudo) prtApplyToPseudo.checked = true;
        if (prtAdaptive) prtAdaptive.checked = true;
        if (prtAdaptiveSettings) prtAdaptiveSettings.style.display = 'block';
        if (prtExclusionZ) prtExclusionZ.value = 2.5;
        if (prtDirectionCheck) prtDirectionCheck.checked = true;
        if (prtDirectionPenalty) prtDirectionPenalty.value = 0.3;
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
