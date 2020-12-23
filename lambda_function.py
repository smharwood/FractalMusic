"""
AWS Lambda function
"""
import base64
from chaos_music import GetParameters, GenerateImage, plotter

def lambda_handler(event, context):
    """
    Form of event:
    {
    "resource": "Resource path",
    "path": "Path parameter",
    "httpMethod": "Incoming request's method name"
    "headers": {String containing incoming request headers}
    "multiValueHeaders": {List of strings containing incoming request headers}
    "queryStringParameters": {query string parameters }
    "multiValueQueryStringParameters": {List of query string parameters}
    "pathParameters":  {path parameters}
    "stageVariables": {Applicable stage variables}
    "requestContext": {Request context, including authorizer-returned key-value pairs}
    "body": "A JSON string of the request payload."
    "isBase64Encoded": "A boolean flag to indicate if the applicable request payload is Base64-encoded"
    }
    """
    # TODO: make use of Matplotlib kosher (see logs)

    name = 'fractal'
    iterations = 5e4
    # Set max iterations higher if not a test
    try:
        req_con = event['requestContext']
        if req_con['stage'] == 'pub':
            iterations = 5e5
    except Exception: pass
    params = event['queryStringParameters']
    # Weight parameters
    wts = []
    for i in range(6):
        try:
            wts.append(float(params['w{}'.format(i)]))
        except Exception: pass
    # use default
    if len(wts) < 3: wts = None
    # Move parameters
    mfs = []
    for i in range(6):
        try:
            mfs.append(float(params['m{}'.format(i)]))
        except Exception: pass
    # use default
    if len(mfs) < 3: mfs = None

    print(event)
    print("Weights: {}".format(wts))
    print("Moves: {}".format(mfs))

    # Generate fractal
    # note that scratch space is at /tmp
    image_name = "/tmp/{}.png".format(name)
    generate_fractal(image_name, iterations, wts, mfs)
    with open(image_name, 'rb') as image:
        byte_string = base64.b64encode(image.read())
        return {
            'headers': { "Content-Type": 'image/png',
                         "Access-Control-Allow-Origin": '*',
                         "Access-Control-Allow-Headers": 'Content-Type, X-Api-Key' },
            'statusCode': 200,
            'body': byte_string.decode('utf-8'),
            'isBase64Encoded': True
        }
    # else
    return {
        'headers': { "Content-Type": 'text/html',
                     "Access-Control-Allow-Origin": '*' },
        'statusCode': 200,
        'body': '<h1>I guess this did not work</h1>',
    }


def generate_fractal(image_name, n_iterations, wts=None, mfs=None):
    basis_pts, wts, mfs = GetParameters(wts=wts, mfs=mfs, twist=0, seed=0)
    density = GenerateImage(basis_pts, wts, mfs, n_Iterations=n_iterations)
    plotter(density, image_name, invert=True)
    return
