
from meta_reviewer import MetaReviewer

def test_correctness():
    reviewer = MetaReviewer()

    # 1. Same category
    assert reviewer.calculate_domain_distance("cs.LG", "cs.AI") == 0.1, "Same category check failed"

    # 2. Known pair (physics, q-fin)
    assert reviewer.calculate_domain_distance("physics", "q-fin") == 0.6, "Known pair (physics, q-fin) failed"
    # Order independence
    assert reviewer.calculate_domain_distance("q-fin", "physics") == 0.6, "Known pair order independence failed"

    # 3. Known pair (cs, q-bio)
    assert reviewer.calculate_domain_distance("cs", "q-bio") == 0.6, "Known pair (cs, q-bio) failed"

    # 4. Unknown pair (Disparate Fields)
    assert reviewer.calculate_domain_distance("quant-ph", "q-fin.CP") == 1.0, "Unknown pair failed"

    print("All correctness tests passed!")

if __name__ == "__main__":
    test_correctness()
