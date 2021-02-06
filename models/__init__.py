from .encoder import build_encoder
from .decoder import build_decoder
from .predictor import build_predictor
from .latent_minimization_ebm import LatentMinimizationEBM


def add_model_specific_args(parser):
    # Components
    parser.add_argument("--encoder_type", default="test", type=str)
    parser.add_argument("--decoder_type", default="test", type=str)
    parser.add_argument("--predictor_type", default="baseline", type=str)

    # Transformer
    parser.add_argument(
        "--hidden_predictor_layers",
        default=6,
        type=int,
        help="Number of layers in hidden predictor",
    )
    parser.add_argument(
        "--hidden_predictor_dim_feedforward",
        default=512,
        type=int,
        help="Number of FFN dimensionality in hidden predictor",
    )
    parser.add_argument(
        "--hidden_predictor_nhead",
        default=8,
        type=int,
        help="Number of attention head in hidden predictor",
    )

    # Latent
    parser.add_argument(
        "--latent_optimizer_type",
        choices=["GD", "LBFGS"],
        default="LBFGS",
        type=str
    )
    parser.add_argument("--latent_size", default=2, type=int)
    parser.add_argument("--no_latent", action="store_true")

    # Lambdas
    parser.add_argument("--lambda_target_prediction", default=1.0, type=float)
    # parser.add_argument("--lambda_decoding_error", default=1.0, type=float)

    return parser


def build_model(args):
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    args.embedding_size = encoder._embedding_size
    predictor = build_predictor(args)
    model = LatentMinimizationEBM(
        encoder,
        decoder,
        predictor,
        args.num_conditional_frames,
        args.latent_size,
        args.no_latent,
        args.latent_optimizer_type,
        args.lambda_target_prediction,
    )
    return model
