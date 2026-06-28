#!/bin/bash
set -e
cd "$(dirname "$0")/data/raw"

# Residential Handbooks
# PDFs
curl -fO https://www.ura.gov.sg/-/media/Corporate/Planning/Master-Plan/MP25WrittenStatement.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Flats-Condominiums/Summary-FC-v2.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Landed-Housing/Summary-Semi-Detached.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Landed-Housing/Summary-Strata-Landed-Housing.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Landed-Housing/Summary-Detached.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Landed-Housing/Summary-Terrace.pdf

# Non-Residential Handbooks
# PDFs
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Commercial/Summary-Commercial.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Industrial/Summary-B1.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Industrial/Summary-BP.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Others/Summary-EI.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Others/Summary-CCI.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Others/Summary-Transport.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Hotel/Summary-Hotel.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Industrial/Summary-B2.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Others/Summary-HMC.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Others/Summary-PW.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Others/Summary-SR.pdf
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Agriculture/Summary-Agriculture.pdf

# Gross Floor Area
# PDFs
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/GFA/Summary_GFA.pdf

# Urban Design
# PDFs
curl -fO https://www.ura.gov.sg/-/media/Corporate/Guidelines/Development-control/Circulars/2025/Dec/dc25-11/DTC/dc25-11_DTC.pdf
