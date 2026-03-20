import numpy as np
import pandas as pd


class TestAvaliarTrigger:
    """Testa a lógica de trigger de retraining sem dependência de Spark."""

    def _get_module(self):
        from monitoring.retrain_trigger import _avaliar_trigger
        return _avaliar_trigger

    def test_historico_insuficiente(self):
        _avaliar_trigger = self._get_module()
        df = pd.DataFrame({
            "competencia":        ["202301"],
            "precision_at_k":     [0.60],
            "auc_pr":             [0.65],
            "n_municipios":       [3300],
            "consolidation_flag": ["consolidado"],
            "simulated":          [True],
            "monitor_date":       ["2026-03-17"],
        })
        resultado = _avaliar_trigger(df, precision_k_threshold=0.55, min_consecutive_below=2)
        assert not resultado["trigger"]
        assert "insuficiente" in resultado["reason"].lower()

    def test_sem_trigger_performance_ok(self):
        _avaliar_trigger = self._get_module()
        df = pd.DataFrame({
            "competencia":        [f"20230{i}" for i in range(1, 7)],
            "precision_at_k":     [0.60, 0.62, 0.59, 0.61, 0.63, 0.60],
            "auc_pr":             [0.65] * 6,
            "n_municipios":       [3300] * 6,
            "consolidation_flag": ["consolidado"] * 6,
            "simulated":          [True] * 6,
            "monitor_date":       ["2026-03-17"] * 6,
        })
        resultado = _avaliar_trigger(df, precision_k_threshold=0.55, min_consecutive_below=2)
        assert not resultado["trigger"]
        assert resultado["n_consecutivas_abaixo"] == 0

    def test_trigger_duas_consecutivas(self):
        _avaliar_trigger = self._get_module()
        df = pd.DataFrame({
            "competencia":        [f"20230{i}" for i in range(1, 7)],
            "precision_at_k":     [0.60, 0.62, 0.59, 0.61, 0.49, 0.48],
            "auc_pr":             [0.65] * 6,
            "n_municipios":       [3300] * 6,
            "consolidation_flag": ["consolidado"] * 6,
            "simulated":          [True] * 6,
            "monitor_date":       ["2026-03-17"] * 6,
        })
        resultado = _avaliar_trigger(df, precision_k_threshold=0.55, min_consecutive_below=2)
        assert resultado["trigger"]
        assert resultado["n_consecutivas_abaixo"] == 2
        assert "degradação" in resultado["reason"].lower()

    def test_sem_trigger_uma_consecutiva(self):
        _avaliar_trigger = self._get_module()
        df = pd.DataFrame({
            "competencia":        [f"20230{i}" for i in range(1, 7)],
            "precision_at_k":     [0.60, 0.62, 0.59, 0.61, 0.63, 0.48],
            "auc_pr":             [0.65] * 6,
            "n_municipios":       [3300] * 6,
            "consolidation_flag": ["consolidado"] * 6,
            "simulated":          [True] * 6,
            "monitor_date":       ["2026-03-17"] * 6,
        })
        resultado = _avaliar_trigger(df, precision_k_threshold=0.55, min_consecutive_below=2)
        assert not resultado["trigger"]
        assert resultado["n_consecutivas_abaixo"] == 1

    def test_trigger_queda_abrupta(self):
        _avaliar_trigger = self._get_module()
        df = pd.DataFrame({
            "competencia":        [f"20230{i}" for i in range(1, 9)],
            "precision_at_k":     [0.65, 0.64, 0.63, 0.65, 0.64, 0.63, 0.65, 0.45],
            "auc_pr":             [0.70] * 8,
            "n_municipios":       [3300] * 8,
            "consolidation_flag": ["consolidado"] * 8,
            "simulated":          [True] * 8,
            "monitor_date":       ["2026-03-17"] * 8,
        })
        resultado = _avaliar_trigger(df, precision_k_threshold=0.55, min_consecutive_below=2)
        assert resultado["trigger"]
        assert "abrupta" in resultado["reason"].lower()


class TestPrecisionAtK:
    """Testa o cálculo de Precision@K."""

    def test_precision_at_k_perfeito(self):
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([1,   1,   1,   0,   0,   0])
        k = 3
        ordem = np.argsort(scores)[::-1]
        top_k = labels[ordem][:k]
        prec = top_k.sum() / k
        assert prec == 1.0

    def test_precision_at_k_zero(self):
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.7])
        labels = np.array([1,   1,   1,   0,   0,   0])
        k = 3
        ordem = np.argsort(scores)[::-1]
        top_k = labels[ordem][:k]
        prec = top_k.sum() / k
        assert prec == 0.0
