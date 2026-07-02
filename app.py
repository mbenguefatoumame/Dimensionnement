"""
Application de dimensionnement de la partie radio NG-RAN 5G
--------------------------------------------------------------
Approche : dimensionnement par la couverture (bilan de liaison)
           + dimensionnement par la capacité (trafic)
           Le nombre de sites retenu = max(couverture, capacité)
"""

import streamlit as st
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# 1. FONCTIONS DE CALCUL (logique métier, indépendante de l'interface)
# ----------------------------------------------------------------------

def bruit_thermique_dbm(bande_passante_mhz: float) -> float:
    """Puissance de bruit thermique en dBm pour une bande passante donnée."""
    k = 1.38e-23  # constante de Boltzmann
    T = 290       # température de référence (K)
    bp_hz = bande_passante_mhz * 1e6
    n_watts = k * T * bp_hz
    return 10 * np.log10(n_watts * 1000)  # conversion en dBm


def sensibilite_recepteur(bande_passante_mhz, noise_figure_db, sinr_cible_db):
    """Sensibilité du récepteur (dBm)."""
    return bruit_thermique_dbm(bande_passante_mhz) + noise_figure_db + sinr_cible_db


def eirp(p_tx_dbm, gain_tx_dbi, pertes_cable_db):
    """Puissance isotrope rayonnée équivalente (dBm)."""
    return p_tx_dbm + gain_tx_dbi - pertes_cable_db


def mapl(eirp_dbm, sensibilite_dbm, gain_rx_dbi, marge_shadow_db,
         marge_interference_db, perte_corps_db, perte_penetration_db):
    """Maximum Allowable Path Loss (dB)."""
    return (eirp_dbm - sensibilite_dbm + gain_rx_dbi
            - marge_shadow_db - marge_interference_db
            - perte_corps_db - perte_penetration_db)


def rayon_cellule_spm(mapl_db, freq_mhz, h_site_m, h_ue_m, k_clutter_db,
                       k1=-13.82, k2=44.9):
    """
    Rayon max de cellule (km) obtenu en inversant le modèle SPM
    (dérivé COST-231 Hata), forme simplifiée :
        L = K1 + K2*log10(h_site) + (44.9 - 6.55*log10(h_site))*log10(d_km)
            + 26.16*log10(f) - 13.82*log10(h_site) - a(h_ue) + K_clutter
    On isole d_km.
    """
    a_hue = (1.1 * np.log10(freq_mhz) - 0.7) * h_ue_m - (1.56 * np.log10(freq_mhz) - 0.8)
    cste = (46.3 + 33.9 * np.log10(freq_mhz) - 13.82 * np.log10(h_site_m)
            - a_hue + k_clutter_db)
    pente = 44.9 - 6.55 * np.log10(h_site_m)
    exposant = (mapl_db - cste) / pente
    d_km = 10 ** exposant
    return max(d_km, 0.01)


def surface_cellule_km2(rayon_km, forme="hexagonale"):
    """Surface couverte par une cellule (approximation hexagonale)."""
    if forme == "hexagonale":
        return 2.6 * rayon_km ** 2
    return np.pi * rayon_km ** 2


def nb_sites_couverture(surface_zone_km2, surface_cellule_km2_val, secteurs_par_site):
    """Nombre de sites nécessaires pour la couverture."""
    surface_site = surface_cellule_km2_val * secteurs_par_site
    return int(np.ceil(surface_zone_km2 / surface_site))


def capacite_cellule_mbps(bande_passante_mhz, efficacite_spectrale_bps_hz, overhead_pct):
    """Débit max par secteur (Mbps)."""
    return bande_passante_mhz * efficacite_spectrale_bps_hz * (1 - overhead_pct / 100)


def nb_sites_capacite(trafic_total_mbps, capacite_secteur_mbps, secteurs_par_site,
                       facteur_charge_pct):
    """Nombre de sites nécessaires pour absorber le trafic."""
    capacite_site = capacite_secteur_mbps * secteurs_par_site * (facteur_charge_pct / 100)
    return int(np.ceil(trafic_total_mbps / capacite_site))


# ----------------------------------------------------------------------
# 2. INTERFACE STREAMLIT
# ----------------------------------------------------------------------

st.set_page_config(page_title="Dimensionnement NG-RAN 5G", layout="wide")
st.title("📡 Dimensionnement de la partie radio NG-RAN 5G")
st.caption("Dimensionnement par couverture (bilan de liaison) et par capacité (trafic)")

with st.sidebar:
    st.header("1. Paramètres radio")
    freq_mhz = st.selectbox("Bande de fréquence (MHz)",
                             [700, 2100, 3500, 26000], index=2)
    bande_passante_mhz = st.number_input("Bande passante (MHz)", value=100.0, min_value=1.0)
    p_tx_dbm = st.number_input("Puissance d'émission par antenne (dBm)", value=43.0)
    gain_tx_dbi = st.number_input("Gain antenne émission (dBi)", value=18.0)
    gain_rx_dbi = st.number_input("Gain antenne UE (dBi)", value=0.0)
    pertes_cable_db = st.number_input("Pertes câbles/connecteurs (dB)", value=2.0)
    noise_figure_db = st.number_input("Facteur de bruit récepteur (dB)", value=7.0)
    sinr_cible_db = st.number_input("SINR cible en bord de cellule (dB)", value=-6.0)

    st.header("2. Marges du bilan de liaison")
    marge_shadow_db = st.number_input("Marge de shadowing (dB)", value=8.0)
    marge_interference_db = st.number_input("Marge d'interférence (dB)", value=3.0)
    perte_corps_db = st.number_input("Perte corps humain (dB)", value=3.0)
    perte_penetration_db = st.number_input("Perte de pénétration bâtiment (dB)", value=15.0)

    st.header("3. Site & environnement")
    h_site_m = st.number_input("Hauteur antenne site (m)", value=30.0)
    h_ue_m = st.number_input("Hauteur UE (m)", value=1.5)
    type_zone = st.selectbox("Type de zone (clutter)",
                              ["Urbain dense", "Urbain", "Suburbain", "Rural"])
    k_clutter_map = {"Urbain dense": 3.0, "Urbain": 0.0,
                      "Suburbain": -5.0, "Rural": -10.0}
    k_clutter_db = k_clutter_map[type_zone]
    secteurs_par_site = st.number_input("Nombre de secteurs par site", value=3, min_value=1, step=1)

    st.header("4. Trafic & capacité")
    surface_zone_km2 = st.number_input("Surface de la zone à couvrir (km²)", value=25.0)
    densite_utilisateurs = st.number_input("Densité d'utilisateurs (users/km²)", value=500.0)
    debit_par_user_mbps = st.number_input("Débit moyen requis par utilisateur (Mbps)", value=10.0)
    efficacite_spectrale = st.number_input("Efficacité spectrale (bit/s/Hz)", value=4.5)
    overhead_pct = st.number_input("Overhead signalisation (%)", value=20.0)
    facteur_charge_pct = st.number_input("Facteur de charge cible (%)", value=70.0)

# ----------------------------------------------------------------------
# 3. CALCULS
# ----------------------------------------------------------------------

if st.button("🚀 Lancer le dimensionnement", type="primary"):

    # --- Dimensionnement par la couverture ---
    sens = sensibilite_recepteur(bande_passante_mhz, noise_figure_db, sinr_cible_db)
    eirp_val = eirp(p_tx_dbm, gain_tx_dbi, pertes_cable_db)
    mapl_val = mapl(eirp_val, sens, gain_rx_dbi, marge_shadow_db,
                     marge_interference_db, perte_corps_db, perte_penetration_db)
    rayon_km = rayon_cellule_spm(mapl_val, freq_mhz, h_site_m, h_ue_m, k_clutter_db)
    surf_cell = surface_cellule_km2(rayon_km)
    sites_couverture = nb_sites_couverture(surface_zone_km2, surf_cell, secteurs_par_site)

    # --- Dimensionnement par la capacité ---
    trafic_total_mbps = surface_zone_km2 * densite_utilisateurs * debit_par_user_mbps
    capacite_secteur = capacite_cellule_mbps(bande_passante_mhz, efficacite_spectrale, overhead_pct)
    sites_capacite = nb_sites_capacite(trafic_total_mbps, capacite_secteur,
                                        secteurs_par_site, facteur_charge_pct)

    # --- Dimensionnement final ---
    sites_final = max(sites_couverture, sites_capacite)
    critere = "Couverture" if sites_couverture >= sites_capacite else "Capacité"

    # ------------------------------------------------------------------
    # 4. AFFICHAGE DES RÉSULTATS
    # ------------------------------------------------------------------
    st.subheader("Résultats du bilan de liaison")
    c1, c2, c3 = st.columns(3)
    c1.metric("Sensibilité récepteur", f"{sens:.1f} dBm")
    c2.metric("EIRP", f"{eirp_val:.1f} dBm")
    c3.metric("MAPL", f"{mapl_val:.1f} dB")

    st.subheader("Dimensionnement par la couverture")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rayon de cellule", f"{rayon_km:.2f} km")
    c2.metric("Surface / site", f"{surf_cell * secteurs_par_site:.2f} km²")
    c3.metric("Sites nécessaires (couverture)", sites_couverture)

    st.subheader("Dimensionnement par la capacité")
    c1, c2, c3 = st.columns(3)
    c1.metric("Trafic total zone", f"{trafic_total_mbps:.0f} Mbps")
    c2.metric("Capacité / secteur", f"{capacite_secteur:.1f} Mbps")
    c3.metric("Sites nécessaires (capacité)", sites_capacite)

    st.subheader("✅ Résultat final")
    st.success(
        f"Nombre de sites à déployer : **{sites_final}** "
        f"(critère dimensionnant : **{critere}**)"
    )

    df_comparatif = pd.DataFrame({
        "Critère": ["Couverture", "Capacité"],
        "Nombre de sites": [sites_couverture, sites_capacite]
    })
    st.bar_chart(df_comparatif.set_index("Critère"))

    with st.expander("📋 Détails des paramètres utilisés"):
        st.json({
            "Fréquence (MHz)": freq_mhz,
            "Bande passante (MHz)": bande_passante_mhz,
            "Type de zone": type_zone,
            "Hauteur site (m)": h_site_m,
            "Secteurs par site": secteurs_par_site,
            "Surface zone (km²)": surface_zone_km2,
        })

else:
    st.info("Renseigne les paramètres dans le menu latéral puis clique sur "
            "**Lancer le dimensionnement**.")
