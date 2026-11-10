from savia.rag.chunking import ChunkingConfig, chunk_document, make_chunk_id
from savia.rag.models import SourceDocument

_BODY = """# Conditions générales de vente

> ⚠️ **Document archivé, non applicable.** Cette version n'est plus en vigueur depuis le 15 janvier 2026. Elle est conservée à titre de référence historique uniquement. Pour toute question actuelle, se référer à **cgv-v3**.
> Elle reste applicable uniquement aux commandes passées entre le 1er mars 2024 et le 14 janvier 2026.

Lumibox SAS propose des box thématiques et des produits du quotidien, vendus exclusivement en ligne à des clients particuliers. Les présentes conditions générales ont été rédigées pour informer le client, avant toute commande, de ses droits et de ses obligations. Elles précisent comment une commande est passée, payée, livrée, puis éventuellement retournée ou remboursée.

Le client est invité à les lire attentivement avant de valider son panier. En cochant la case d'acceptation lors du paiement, il reconnaît en avoir pris connaissance et les accepter sans réserve. Lumibox SAS conserve la trace de cette acceptation pendant toute la durée légale de conservation des contrats conclus à distance.

Les photographies et descriptions présentes sur le site n'ont qu'une valeur indicative. Les couleurs, les emballages et la composition exacte d'une box peuvent légèrement varier d'un mois à l'autre, notamment lorsqu'un fournisseur modifie la présentation d'un produit. Ces variations ne constituent pas un défaut de conformité tant que la valeur et l'esprit de la box sont respectés.

Lumibox SAS s'efforce de garantir la disponibilité des produits affichés. En cas de rupture de stock après la commande, le client en est informé par courriel dans les meilleurs délais. Il peut alors choisir entre un produit de remplacement de valeur équivalente ou l'annulation de la ligne concernée, avec remboursement du montant correspondant.

Le service client est joignable par courriel du lundi au vendredi, de 9 h à 18 h, hors jours fériés. Toute réclamation doit mentionner le numéro de commande et, le cas échéant, être accompagnée de photographies du produit concerné. Une réponse est apportée dans un délai moyen de 48 heures ouvrées.

## Article 1 — Objet
Les présentes CGV régissent la vente de box et de produits par Lumibox SAS aux consommateurs.

## Article 2 — Commande et prix
Prix en euros TTC, hors frais de livraison.

## Article 3 — Livraison
Délais indicatifs de 3 à 5 jours ouvrés en France métropolitaine.

## Article 4 — Délai de remboursement et de retour
Le client disposait d'un délai de **14 jours après la livraison** pour demander un retour ou un remboursement. Passé ce délai, aucune demande n'était acceptée.

## Article 5 — Remboursement
Les remboursements étaient tous traités par un conseiller, sans plafond automatique. Aucune règle de limite de fréquence n'existait.

## Article 6 — Abonnements
Résiliation avec préavis de **10 jours** avant la date de prélèvement. Aucune pause n'était prévue.

## Article 7 — Médiation
Le consommateur pouvait saisir le médiateur de la consommation désigné aux CGV.
"""

_DOCUMENT = SourceDocument(
    doc_id="policy-refund",
    title="Politique de remboursement",
    category="policy",
    version="2023-01",
    effective_date="2023-01-03",
    status="active",
    body=_BODY,
)

_CONFIG = ChunkingConfig()


def test_make_chunk_id() -> None:
    chunk_id = make_chunk_id("policy", "2026-12", 12, _CONFIG)
    assert chunk_id == "policy@2026-12#012/" + _CONFIG.version


def test_chunking_size() -> None:
    chunks = chunk_document(_DOCUMENT, _CONFIG)
    assert chunks is not None
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.text) <= _CONFIG.chunk_size


def test_chunking_empty_chunk() -> None:
    chunks = chunk_document(_DOCUMENT, _CONFIG)
    assert chunks is not None
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.text.strip() != ""


def test_chunking_deterministic() -> None:
    assert chunk_document(_DOCUMENT, _CONFIG) == chunk_document(_DOCUMENT, _CONFIG)
