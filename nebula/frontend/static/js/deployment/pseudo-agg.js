// Pseudo Aggregation Module
const PseudoAggManager = (function() {
    const Settings = {
        enabled: false,
        emaAlpha: 0.25,
        weightDropRate: 1.0,
        weightScheduleStep: 1,
        stopPseudoRound: null
    };

    function initialize() {
        const pseudoAggSwitch = document.getElementById('pseudoAggSwitch');
        const pseudoAggSettings = document.getElementById('pseudoAggSettings');
        const emaAlphaInput = document.getElementById('emaAlphaInput');
        const emaAlphaValue = document.getElementById('emaAlphaValue');
        const weightDropRate = document.getElementById('weightDropRate');
        const weightScheduleStep = document.getElementById('weightScheduleStep');
        const stopPseudoRound = document.getElementById('stopPseudoRound');

        if (!pseudoAggSwitch || !pseudoAggSettings || !emaAlphaInput || !emaAlphaValue) {
            console.warn('Pseudo Aggregation controls not found in DOM');
            return;
        }

        // Toggle settings visibility when switch is changed
        pseudoAggSwitch.addEventListener('change', function() {
            Settings.enabled = pseudoAggSwitch.checked;
            pseudoAggSettings.style.display = pseudoAggSwitch.checked ? 'block' : 'none';
            console.log('Pseudo Aggregation enabled:', Settings.enabled);
        });

        // Sync slider with number input
        emaAlphaInput.addEventListener('input', function() {
            const value = parseFloat(emaAlphaInput.value) / 100; // Convert 0-100 to 0.0-1.0
            emaAlphaValue.value = value.toFixed(2);
            Settings.emaAlpha = value;
            console.log('EMA Alpha updated:', Settings.emaAlpha);
        });

        // Sync number input with slider
        emaAlphaValue.addEventListener('input', function() {
            const value = parseFloat(emaAlphaValue.value);
            if (value >= 0 && value <= 1) {
                emaAlphaInput.value = Math.round(value * 100); // Convert 0.0-1.0 to 0-100
                Settings.emaAlpha = value;
                console.log('EMA Alpha updated:', Settings.emaAlpha);
            }
        });

        // Weight drop rate listener
        if (weightDropRate) {
            weightDropRate.addEventListener('input', function() {
                Settings.weightDropRate = parseFloat(weightDropRate.value);
                console.log('Weight Drop Rate updated:', Settings.weightDropRate);
            });
        }

        // Weight schedule step listener
        if (weightScheduleStep) {
            weightScheduleStep.addEventListener('input', function() {
                Settings.weightScheduleStep = parseInt(weightScheduleStep.value) || 1;
                console.log('Weight Schedule Step updated:', Settings.weightScheduleStep);
            });
        }

        // Stop pseudo round listener
        if (stopPseudoRound) {
            stopPseudoRound.addEventListener('input', function() {
                Settings.stopPseudoRound = stopPseudoRound.value ? parseInt(stopPseudoRound.value) : null;
                console.log('Stop Pseudo Round updated:', Settings.stopPseudoRound);
            });
        }

        console.log('Pseudo Aggregation controls initialized');
    }

    function getSettings() {
        return {
            enabled: Settings.enabled,
            ema_alpha: Settings.emaAlpha,
            weight_drop_rate: Settings.weightDropRate,
            weight_schedule_step: Settings.weightScheduleStep,
            stop_pseudo_round: Settings.stopPseudoRound
        };
    }

    function isEnabled() {
        return Settings.enabled;
    }

    function getEmaAlpha() {
        return Settings.emaAlpha;
    }

    return {
        initialize,
        getSettings,
        isEnabled,
        getEmaAlpha
    };
})();

export default PseudoAggManager;
