
class TestAbRoute:
    """Testa o roteamento A/B determinístico."""

    def _get_fn(self):
        from scoring.batch_score import _ab_route
        return _ab_route

    def test_sem_challenger_sempre_champion(self):
        _ab_route = self._get_fn()
        for municipio in ["110001", "355030", "999999", "000001"]:
            assert _ab_route(municipio, challenger_exists=False, ab_challenger_pct=0.20) == "champion"

    def test_com_challenger_deterministico(self):
        _ab_route = self._get_fn()
        # mesmo municipio_id sempre retorna o mesmo resultado
        resultado1 = _ab_route("355030", challenger_exists=True, ab_challenger_pct=0.20)
        resultado2 = _ab_route("355030", challenger_exists=True, ab_challenger_pct=0.20)
        assert resultado1 == resultado2

    def test_com_challenger_distribuicao_aproximada(self):
        _ab_route = self._get_fn()
        # com 1000 IDs sintéticos, ~20% deve ir para challenger
        resultados = [
            _ab_route(str(i).zfill(6), challenger_exists=True, ab_challenger_pct=0.20)
            for i in range(1000)
        ]
        pct_challenger = resultados.count("challenger") / len(resultados)
        # tolerância de ±5pp em torno de 20%
        assert 0.15 <= pct_challenger <= 0.25
